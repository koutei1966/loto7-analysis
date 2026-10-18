from itertools import combinations
from math import comb

print("ロト7 全件分析開始")
print("全組み合わせ数:", comb(37, 7))

count = 0
for numbers in combinations(range(1, 38), 7):
    count += 1

print("全件検査完了:", count)
