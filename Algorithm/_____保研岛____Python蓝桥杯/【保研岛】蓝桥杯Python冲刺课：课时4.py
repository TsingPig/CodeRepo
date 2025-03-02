import sys
input = lambda: sys.stdin.readline().strip()
from math import *
def is_prime(n):
    if n < 2:
        return False
    # 121 = 11 * 11，最大的因子不会超过 sqrt(n)
    for i in range(2, sqrt(n) + 1):  # 只需检查到 sqrt(n)
        if n % i == 0:
            return False
    return True
n = int(input())
print(is_prime(n))

