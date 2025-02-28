'''
【保研岛】蓝桥杯Python辅导课：课时3_2024_CA_省C.训练士兵（注释版）
'''
# 快读模板，可以提高读入数据效率，建议熟练默写
import sys
input = lambda:sys.stdin.readline().strip()


'''
lambda 参数: 表达式
函数变量 = lambda 参数: 表达式
定义一个匿名函数，并赋值给x
函数拥有一个参数a，输出a + 10
'''
x = lambda a: a + 10
print(x(5)) # 输出: 15

get_mid = lambda nums: nums[len(nums) / 2]
print(get_mid([1, 4, 5))


# 数据读入
n, S = map(int, input().split())
nums = [[0, 0]] * n # 用于排序
p, c = [0] * n, [0] * n

'''
思考：
显然团购价是一直不变的，有些人训练的次数需要的少，有些人多
可以在大家都需要团购的时候先团购，当团购不合适的时候，再单独训练
——联想到贪心 + 排序（按照训练次数由低到高）

res最终花费
tot所有人单独训练一次成本
cnt当前“团购”了的次数

思路：根据训练次数由低到高排序，对训练次数进行遍历：
if S <= tot: # 团购合适
else:  # 当前团购不合适，剩下的人单独训练
'''


# 数据预处理
for i in range(n):
    nums[i] = list(map(int, input().split()))
# 排序：根据nums[i][1]即次数排序，默认是由低到高
nums.sort(key = lambda x: x[1])
for i in range(n):
    p[i], c[i] = nums[i][0], nums[i][1]

res = cnt = 0
tot = sum(p)
for i in range(n):
    if tot >= S:    # 团购合适
        res += (c[i] - cnt) * S
        cnt += c[i] - cnt
    else:   # 团购不合适，剩下的人单独训练
        res += (c[i] - cnt) * p[i]
    tot -= p[i] # 第i人完成训练，减去他的单独训练成本
print(res)







    
