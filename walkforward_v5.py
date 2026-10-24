
"""Loto7 v5: 3% walk-forward structural filter and independent-run co-occurrence.
Preserves v2-v4 and uses only historical draws to select three tickets.
"""
import csv
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

OUT = Path('results')
OUT.mkdir(exist_ok=True)
FEATURES = ('odd_count', 'sum_bucket', 'adjacent_pairs', 'triple_run', 'carry_count', 'largest_band')
WEIGHTS = (
    (1.4, 1.5, .24, .15, .15, .16, .12),
    (.8, 2.3, .4, .2, .10, .25, .18),
    (2.0, .9, .2, .2, .30, .08, .30),
)


def load():
    with open('draws.csv', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not all(f'n{i}' in (reader.fieldnames or []) for i in range(1, 8)):
            raise ValueError('Missing n1..n7')
        draws = [tuple(sorted(int(row[f'n{i}']) for i in range(1, 8))) for row in reader]
    if len(draws) < 21 or any(len(set(d)) != 7 or d[0] < 1 or d[-1] > 37 for d in draws):
        raise ValueError('Invalid draw history')
    return draws


def patterns(c, previous):
    bands = [sum(a <= x <= b for x in c) for a, b in ((1, 12), (13, 24), (25, 37))]
    return (sum(x % 2 for x in c), sum(c)//15,
            sum(b == a+1 for a, b in zip(c, c[1:])),
            int(any(c[i+2] == c[i]+2 for i in range(5))),
            len(set(c) & set(previous)), max(bands))


def run_signature(c):
    runs = []
    start = previous = c[0]
    for x in c[1:]:
        if x != previous + 1:
            if previous > start:
                runs.append(previous - start + 1)
            start = x
        previous = x
    if previous > start:
        runs.append(previous - start + 1)
    return tuple(sorted(runs, reverse=True))


def run_starts(c):
    """Return (start, length) for each maximal independent consecutive run.

    A triple 02-03-04 is one run (2, 3), not two overlapping pairs.
    """
    runs = []
    start = previous = c[0]
    for x in c[1:]:
        if x != previous + 1:
            if previous > start:
                runs.append((start, previous - start + 1))
            start = x
        previous = x
    if previous > start:
        runs.append((start, previous - start + 1))
    return tuple(runs)


def rules(history):
    counts = [Counter() for _ in FEATURES]
    structures = Counter()
    for i in range(1, len(history)):
        for j, value in enumerate(patterns(history[i], history[i-1])):
            counts[j][value] += 1
        structures[run_signature(history[i])] += 1
    n = len(history) - 1
    # Strictly exclude rates below 3%; equality survives.
    keep = [set(v for v, count in c.items() if 100 * count >= 3 * n) for c in counts]
    keep_structures = {v for v, count in structures.items() if 100 * count >= 3 * n}
    return keep, keep_structures


def context(history):
    n = len(history)
    freq = Counter(x for d in history for x in d)
    recent = Counter(x for d in history[-30:] for x in d)
    pairs, triples = Counter(), Counter()
    for d in history[-120:]:
        for i, a in enumerate(d):
            for j in range(i+1, 7):
                b = d[j]
                pairs[a, b] += 1
                for k in range(j+1, 7):
                    triples[a, b, d[k]] += 1
    gaps = {x: n for x in range(1, 38)}
    seen = set()
    for i, d in enumerate(reversed(history)):
        for x in d:
            if x not in seen:
                gaps[x] = i
                seen.add(x)
    run_counts = Counter()
    run_pairs = Counter()
    for d in history[-120:]:
        starts = run_starts(d)
        run_counts.update(starts)
        run_pairs.update(combinations(starts, 2))
    return freq, recent, pairs, triples, gaps, set(history[-1]), run_counts, run_pairs


def score(c, ctx, w, n):
    freq, recent, pairs, triples, gaps, last, run_counts, run_pairs = ctx
    f, r, carry, shift, gap, pair, triple = w
    val = 0.0
    for x in c:
        distance = min(abs(x-y) for y in last)
        val += (f*freq[x]/n + r*recent[x]/min(30, n)
                + carry*(x in last) + shift*(distance in (1, 2))
                + gap*min(gaps[x], 15)/15)
    window = min(120, n)
    val += pair*sum(pairs[a, b] for i, a in enumerate(c) for b in c[i+1:])/window
    val += triple*sum(triples[a, b, d] for i, a in enumerate(c)
                      for j, b in enumerate(c[i+1:], start=i+1)
                      for d in c[j+1:])/window
    # Independent 2/3/4+ number runs and their joint occurrence.
    starts = run_starts(c)
    val += .10 * sum(run_counts[r] for r in starts) / window
    val += .18 * sum(run_pairs[p] for p in combinations(starts, 2)) / window
    return val


def candidates(history, keep_and_structures):
    keep, keep_structures = keep_and_structures
    rng = random.Random(314159 + len(history)*97)
    previous = history[-1]
    pool, rejected = set(), Counter()
    freq = Counter(x for d in history for x in d)
    recent = Counter(x for d in history[-30:] for x in d)
    ranked = sorted(range(1, 38), key=lambda x: (-freq[x]/len(history)-recent[x]/min(30, len(history)), x))
    for i in range(2400):
        group = ranked[:rng.choice((16, 23, 30, 37))] if i % 2 else range(1, 38)
        c = tuple(sorted(rng.sample(group, 7)))
        failed = [j for j, v in enumerate(patterns(c, previous)) if v not in keep[j]]
        if run_signature(c) not in keep_structures:
            rejected["run_structure"] += 1
        elif failed:
            rejected.update(failed)
        else:
            pool.add(c)
    if len(pool) < 3:
        raise RuntimeError('Fewer than three candidates survive strict filters')
    return pool, rejected


def choose(history, past_hits):
    pool, rejected = candidates(history, rules(history))
    ctx = context(history)
    recent = past_hits[-60:]
    avg = [sum(row[j] for row in recent)/len(recent) if recent else 0 for j in range(3)]
    order = sorted(range(3), key=lambda j: (-avg[j], j))
    picks = []
    for j in order:
        ranking = sorted(pool, key=lambda c: (-score(c, ctx, WEIGHTS[j], len(history)), c))
        selected = next((c for c in ranking if all(len(set(c) & set(p)) <= 4 for p in picks)), None)
        if selected is None:
            raise RuntimeError('Cannot select three diverse tickets')
        picks.append(selected)
    return picks, order, rejected, len(pool)


def main():
    draws = load()
    rows, performance = [], []
    rejected_total = Counter()
    for i in range(20, len(draws)):
        picks, order, rejected, pool_count = choose(draws[:i], performance)
        hits = [len(set(p) & set(draws[i])) for p in picks]
        by_strategy = [0, 0, 0]
        for position, strategy in enumerate(order):
            by_strategy[strategy] = hits[position]
        performance.append(by_strategy)
        rejected_total.update(rejected)
        rows.append([i+1, *['-'.join(map(str, p)) for p in picks], *hits,
                     max(hits), order[0]+1, pool_count])
    with (OUT/'walkforward_v5.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['draw_index','ticket_1','ticket_2','ticket_3','hits_1','hits_2','hits_3',
                         'best_hits','best_strategy','candidate_count'])
        writer.writerows(rows)
    picks, order, rejected, pool_count = choose(draws, performance)
    summary = {
        'model': 'conditional_runs_v5', 'draws_loaded': len(draws), 'tests': len(rows),
        'average_best_hits': round(sum(r[7] for r in rows)/len(rows), 4),
        'draw_700_validation': next((r for r in rows if r[0] == 700), None),
        'next_draw': len(draws)+1, 'next_three_tickets': [list(p) for p in picks],
        'strategy_order': [j+1 for j in order],
        'filter_rejections_all_backtests': {**{FEATURES[k]: rejected_total[k] for k in range(6)}, 'run_structure': rejected_total['run_structure']},
        'filter_rejections_next_draw': {**{FEATURES[k]: rejected[k] for k in range(6)}, 'run_structure': rejected['run_structure']},
        'next_candidate_count': pool_count,
        'limitations': 'Sampled candidates; month/date effects and 4-7 number co-occurrence not yet included. Strict 3% historical filter may remove rare patterns.'
    }
    (OUT/'summary_v5.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
