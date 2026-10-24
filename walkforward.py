import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

DATA = Path('draws.csv')
OUT = Path('results')
OUT.mkdir(exist_ok=True)
NUMBERS = range(1, 38)


def load_draws():
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        keys = [f'n{i}' for i in range(1, 8)]
        if not all(k in (reader.fieldnames or []) for k in keys):
            raise ValueError('Missing n1..n7 columns')
        draws = []
        for row in reader:
            d = tuple(sorted(int(row[k]) for k in keys))
            if len(set(d)) != 7 or d[0] < 1 or d[-1] > 37:
                raise ValueError('Invalid draw')
            draws.append(d)
    if len(draws) < 21:
        raise ValueError('At least 21 draws required')
    return draws


def features(history):
    total = len(history)
    freq = Counter(n for d in history for n in d)
    recent = Counter(n for d in history[-30:] for n in d)
    pair = Counter()
    for d in history[-120:]:
        for i, a in enumerate(d):
            for b in d[i+1:]:
                pair[(a, b)] += 1
    last = set(history[-1])
    age = {n: total for n in NUMBERS}
    for j, d in enumerate(reversed(history)):
        for n in d:
            if age[n] == total:
                age[n] = j
    base = {}
    for n in NUMBERS:
        shifted = min((abs(n - p) for p in last), default=37)
        base[n] = (1.4 * freq[n] / total +
                   1.5 * recent[n] / min(30, total) +
                   0.24 * (n in last) +
                   0.15 * (shifted == 1) +
                   0.08 * (shifted == 2) +
                   0.15 * min(age[n], 15) / 15)
    return base, pair, last


def score(combo, base, pair, last, history):
    c = tuple(sorted(combo))
    odds = sum(n % 2 for n in c)
    total = sum(c)
    adjacent = sum(b == a + 1 for a, b in zip(c, c[1:]))
    triple = any(c[i+2] == c[i]+2 for i in range(5))
    bands = [sum(lo <= n <= hi for n in c) for lo, hi in ((1, 12), (13, 24), (25, 37))]
    carry = len(set(c) & last)
    pair_score = sum(pair[(a, b)] for i, a in enumerate(c) for b in c[i+1:])
    return (sum(base[n] for n in c) + 0.016 * pair_score / max(1, min(120, len(history)))
            - 0.17 * abs(odds - 3.5) - 0.008 * abs(total - 133)
            - 0.16 * abs(adjacent - 1) - 0.25 * triple
            - 0.16 * sum(max(0, b - 3) for b in bands)
            - 0.13 * abs(carry - 1))


def predict_three(history):
    base, pair, last = features(history)
    rng = random.Random(1729 + len(history))
    ranked = sorted(NUMBERS, key=lambda n: (-base[n], n))
    candidates = {tuple(sorted(ranked[:7]))}
    for t in (0.09, 0.18, 0.32):
        for _ in range(260):
            remaining = list(NUMBERS)
            selected = []
            for _ in range(7):
                weights = [math.exp((base[n] - max(base.values())) / t) for n in remaining]
                chosen = rng.choices(remaining, weights=weights, k=1)[0]
                selected.append(chosen)
                remaining.remove(chosen)
            candidates.add(tuple(sorted(selected)))
    scored = sorted(candidates, key=lambda c: (-score(c, base, pair, last, history), c))
    selected = []
    for c in scored:
        if all(len(set(c) & set(other)) <= 4 for other in selected):
            selected.append(c)
            if len(selected) == 3:
                break
    if len(selected) != 3:
        raise RuntimeError('Could not select three combinations')
    return selected


def main():
    draws = load_draws()
    results = []
    distributions = [Counter(), Counter(), Counter()]
    for i in range(20, len(draws)):
        picks = predict_three(draws[:i])
        actual = set(draws[i])
        hits = [len(actual & set(p)) for p in picks]
        for j, h in enumerate(hits):
            distributions[j][h] += 1
        results.append([i + 1, *['-'.join(map(str, p)) for p in picks], *hits, max(hits)])
    with (OUT / 'walkforward_3tickets.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['draw_index', 'ticket_1', 'ticket_2', 'ticket_3', 'hits_1', 'hits_2', 'hits_3', 'best_hits'])
        writer.writerows(results)
    next_picks = predict_three(draws)
    summary = {
        'model': 'multi_feature_v1',
        'draws_loaded': len(draws),
        'tests': len(results),
        'average_best_hits': round(sum(r[-1] for r in results) / len(results), 4),
        'per_ticket_hit_distributions': [dict(sorted(d.items())) for d in distributions],
        'draw_700_validation': next((dict(zip(['draw_index', 'ticket_1', 'ticket_2', 'ticket_3', 'hits_1', 'hits_2', 'hits_3', 'best_hits'], r)) for r in results if r[0] == 700), None),
        'next_draw': len(draws) + 1,
        'next_three_tickets': [list(p) for p in next_picks],
        'limitations': 'Initial multi-feature model; 5-percent pattern exclusions and date-conditioned validation are not yet implemented.'
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
