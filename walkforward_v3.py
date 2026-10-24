"""Loto7 adaptive walk-forward model v3. Standard library only."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

OUT = Path('results')
OUT.mkdir(exist_ok=True)
WEIGHTS = ((1.4, 1.5, .24, .15, .15, .16), (.8, 2.3, .4, .2, .10, .25), (2., .9, .2, .2, .30, .08))


def load():
    with open('draws.csv', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not all(f'n{i}' in (reader.fieldnames or []) for i in range(1, 8)):
            raise ValueError('Missing n1..n7')
        draws = [tuple(sorted(int(row[f'n{i}']) for i in range(1, 8))) for row in reader]
    if len(draws) < 21 or any(len(set(d)) != 7 or d[0] < 1 or d[-1] > 37 for d in draws):
        raise ValueError('Invalid draws')
    return draws


def patterns(c, prev):
    bands = [sum(a <= n <= b for n in c) for a, b in ((1, 12), (13, 24), (25, 37))]
    return (sum(n % 2 for n in c), sum(c)//15,
            sum(b == a+1 for a, b in zip(c, c[1:])),
            int(any(c[i+2] == c[i]+2 for i in range(5))),
            len(set(c) & set(prev)), max(bands))


def rules(history):
    counts = [Counter() for _ in range(6)]
    for i in range(1, len(history)):
        for j, value in enumerate(patterns(history[i], history[i-1])):
            counts[j][value] += 1
    n = len(history)-1
    # >=5% historical empirical occurrence, no future information.
    return [set(v for v, count in c.items() if count*20 >= n) for c in counts]


def context(history):
    n = len(history)
    freq = Counter(x for d in history for x in d)
    recent = Counter(x for d in history[-30:] for x in d)
    pairs = Counter()
    for d in history[-120:]:
        for i, a in enumerate(d):
            for b in d[i+1:]:
                pairs[a, b] += 1
    gaps = {x: n for x in range(1, 38)}
    seen = set()
    for i, d in enumerate(reversed(history)):
        for x in d:
            if x not in seen:
                gaps[x] = i
                seen.add(x)
    return freq, recent, pairs, gaps, set(history[-1])


def grade(c, ctx, w, n):
    freq, recent, pairs, gaps, last = ctx
    a, b, carry, shift, gap, pair = w
    result = 0.
    for x in c:
        distance = min(abs(x-y) for y in last)
        result += (a*freq[x]/n + b*recent[x]/min(30, n)
                   + carry*(x in last) + shift*(distance in (1, 2))
                   + gap*min(gaps[x], 15)/15)
    result += pair*sum(pairs[x, y] for i, x in enumerate(c) for y in c[i+1:])/min(120, n)
    return result


def candidates(history, keep):
    rng = random.Random(314159 + len(history)*97)
    last = history[-1]
    pool = set()
    rejected = Counter()
    freq = Counter(x for d in history for x in d)
    recent = Counter(x for d in history[-30:] for x in d)
    ranked = sorted(range(1, 38), key=lambda x: (-freq[x]/len(history)-recent[x]/min(30, len(history)), x))
    for j in range(2400):
        group = ranked[:rng.choice((16, 23, 30, 37))] if j % 2 else range(1, 38)
        c = tuple(sorted(rng.sample(list(group), 7)))
        p = patterns(c, last)
        bad = [k for k, v in enumerate(p) if v not in keep[k]]
        if bad:
            for k in bad:
                rejected[k] += 1
        else:
            pool.add(c)
    if len(pool) < 3:
        raise RuntimeError('Not enough candidates after filtering')
    return pool, rejected


def choose(history, past_hits):
    keep = rules(history)
    pool, rejected = candidates(history, keep)
    ctx = context(history)
    n = len(history)
    recent = past_hits[-60:]
    averages = [sum(row[j] for row in recent)/len(recent) if recent else 0 for j in range(3)]
    order = sorted(range(3), key=lambda j: (-averages[j], j))
    scores = {j: sorted(pool, key=lambda c: (-grade(c, ctx, WEIGHTS[j], n), c)) for j in range(3)}
    picks = []
    # Prefer the historically strongest model first, with ticket diversity.
    for j in order:
        candidate = next((c for c in scores[j] if all(len(set(c) & set(p)) <= 4 for p in picks)), None)
        if candidate is None:
            raise RuntimeError('Could not find sufficiently diverse tickets')
        picks.append(candidate)
    return picks, order, rejected, len(pool)


def main():
    draws = load()
    rows = []
    # Each historical score belongs to the strategy chosen at that draw.
    strategy_hits = []
    rejection_totals = Counter()
    for i in range(20, len(draws)):
        picks, order, rejected, pool_size = choose(draws[:i], strategy_hits)
        actual = set(draws[i])
        hits = [len(actual & set(p)) for p in picks]
        by_strategy = [0]*3
        for pos, strategy in enumerate(order):
            by_strategy[strategy] = hits[pos]
        strategy_hits.append(by_strategy)
        rejection_totals.update(rejected)
        rows.append([i+1, *['-'.join(map(str, p)) for p in picks], *hits, max(hits), order[0]+1, pool_size])
    with (OUT/'walkforward_v3.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['draw_index','ticket_1','ticket_2','ticket_3','hits_1','hits_2','hits_3','best_hits','best_strategy','candidate_count'])
        writer.writerows(rows)
    picks, order, rejected, pool_size = choose(draws, strategy_hits)
    summary = {
        'model': 'adaptive_diversity_v3', 'draws_loaded': len(draws), 'tests': len(rows),
        'average_best_hits': round(sum(row[7] for row in rows)/len(rows), 4),
        'draw_700_validation': next((row for row in rows if row[0] == 700), None),
        'next_draw': len(draws)+1, 'next_three_tickets': [list(p) for p in picks],
        'selected_strategy_order': [j+1 for j in order],
        'filter_rejections_all_backtests': dict(rejection_totals),
        'filter_rejections_next_draw': dict(rejected),
        'next_candidate_count': pool_size,
        'limitations': 'Sampled candidates, not exhaustive; month/date and higher-order co-occurrence not yet implemented.'
    }
    (OUT/'summary_v3.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
