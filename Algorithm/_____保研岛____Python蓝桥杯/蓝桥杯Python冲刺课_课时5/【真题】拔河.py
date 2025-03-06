from math import *
import sys
input = lambda: sys.stdin.readline().strip()
n = int(input())
a = list(map(int, input().split()))

# p[i] = sum(a[:i]), p[i]表示前i个数的和
p = [0] * (n + 1)
for i in range(n):
    p[i + 1] = p[i] + a[i]
    
# s[i] = sum(a[i:]), s[i]表示后(n - i)个数的和
s = [0] * (n + 1)
for i in range(n - 1, -1, -1):
    s[i] = s[i + 1] + a[i]

# 考虑 [0, l)的和 p[l]
# l \in [1, n]
# 考虑 [r, n)的和 s[r]
# r \in [l, n - 1]
res = inf
for l in range(1, n + 1):
    for r in range(l, n):
        res = min(res, abs(p[l] - s[r]))
print(res)
