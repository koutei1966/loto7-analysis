
import itertools
import csv
from collections import Counter
from pathlib import Path

# ロト7 全組み合わせ検証
NUMBERS = range(1, 38)
TOTAL = 0
PASSED = 0

def evaluate(combo):
    odd = sum(n % 2 for n in combo)
    total = sum(combo)
    consecutive = sum(
        b == a + 1
        for a, b in zip(combo, combo[1:])
    )

    if not 2 <= odd <= 5:
        return False
    if not 80 <= total <= 190:
        return False
    if consecutive == 0:
        return False
    if any(
        combo[i+2] == combo[i] + 2
        for i in range(5)
    ):
        return False
    return True

def main():
    global TOTAL, PASSED
    output = Path("analysis_results.csv")

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["numbers"])

        for combo in itertools.combinations(NUMBERS, 7):
            TOTAL += 1
            if evaluate(combo):
                PASSED += 1
                writer.writerow([
                    "-".join(f"{n:02d}" for n in combo)
                ])

    print("検査総数:", TOTAL)
    print("条件通過:", PASSED)
    print("結果:", output)

if __name__ == "__main__":
    main()
