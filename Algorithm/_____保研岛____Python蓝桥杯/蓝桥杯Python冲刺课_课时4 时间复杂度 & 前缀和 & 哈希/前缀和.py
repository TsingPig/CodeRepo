'''
https://www.luogu.com.cn/problem/P8218
'''
import sys
input = lambda: sys.stdin.readline().strip()

n = int(input())
a = list(map(int, input().split()))
q = int(input())

for _ in range(q):
    l, r = map(int, input().split())

    res = 0
    for i in range(l-1, r):
        res += a[i]
    print(res)

    #print(sum(a[l-1:r]))
