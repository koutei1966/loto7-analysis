import csv
import json
from pathlib import Path
from collections import Counter

DATA = Path("draws.csv")
OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)

def load_draws():
    if not DATA.exists():
        raise FileNotFoundError("draws.csv がありません")
    draws = []
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = [f"n{i}" for i in range(1, 8)]
        if not all(k in (reader.fieldnames or []) for k in required):
            raise ValueError("CSVには n1〜n7 の列が必要です")
        for row in reader:
            nums = sorted(int(row[k]) for k in required)
            if len(set(nums)) != 7 or not all(1 <= n <= 37 for n in nums):
                raise ValueError("抽選数字に不正な値があります")
            draws.append(nums)
    if len(draws) < 21:
        raise ValueError("21回以上の履歴が必要です")
    return draws

def predict(history):
    counts = Counter(n for draw in history for n in draw)
    ranked = sorted(range(1, 38), key=lambda n: (-counts[n], n))
    return sorted(ranked[:7])

def main():
    draws = load_draws()
    rows = []
    for i in range(20, len(draws)):
        prediction = predict(draws[:i])
        actual = draws[i]
        hits = len(set(prediction) & set(actual))
        rows.append([i + 1, "-".join(map(str, prediction)), "-".join(map(str, actual)), hits])

    with (OUTPUT / "walkforward.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["draw_index", "prediction", "actual", "hits"])
        writer.writerows(rows)

    summary = {
        "draws_loaded": len(draws),
        "tests": len(rows),
        "average_hits": round(sum(r[3] for r in rows) / len(rows), 4),
        "hit_distribution": dict(sorted(Counter(r[3] for r in rows).items()))
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
