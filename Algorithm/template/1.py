a = [[80, 78, 73, 61, 84, 90],
     [75, 70, 87, 93, 85, 80, 85],
     [76, 94, 81, 81, 88],
     [80, 86, 61, 84, 98, 99],
     [81, 95, 95, 96],
     [90, 94, 95, 96]]

af = [[0.5, 2, 1, 1, 3, 1],
      [2, 1, 1, 3, 3, 1, 1],
      [2, 1, 3, 3, 1],
      [2, 1, 1, 2, 3, 3],
      [2, 3, 1, 3],
      [2, 2, 3, 4]]

b = [[99, 91, 91],
     [100, 93, 74, 99, 94, 95],
     [82, 97, 95, 99, 95, 96],
     [94, 95, 96, 99, 97],
     [89, 95, 96, 95, 98, 95],
     [95, 91, 87, 76]]

bf = [[6, 3, 4],
      [6, 3, 1, 4, 3, 1],
      [3, 2, 1, 4, 1, 3.5],
      [4, 1, 3.5, 3, 3.5],
      [3.5, 4.5, 4, 1, 3.5, 1],
      [1, 1, 4, 3.5]]


c = [25.61, 15.1, 25.06, 25.6, 20.5, 24.7]
# 计算 a 列表的加权平均值
resa = sum(sum(x * xf for x, xf in zip(sublist_a, sublist_af)) / sum(sublist_af) for sublist_a, sublist_af in zip(a, af)) / len(a)

# 计算 b 列表的加权平均值
resb = sum(sum(x * xf for x, xf in zip(sublist_b, sublist_bf)) / sum(sublist_bf) for sublist_b, sublist_bf in zip(b, bf)) / len(b)

# 组合 resa 和 resb
res = resa * 0.2 + resb * 0.8
ex = 2.6
comp = sum(c) / (6 * 30) * 10

print(res * 0.85)
print(ex)
print(comp)
print(res * 0.85 + ex + comp)



