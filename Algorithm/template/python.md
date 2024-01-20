[TOC]

# 列表

## int 转 list

```python
num = 123
nums = list(map(int, str(num)))
```

## list(int) 转 int

```python
nums = [1, 2, 3]
num = int(''.join(map(str, nums)))

def lst_int(nums):
    return int(''.join(map(str, nums)))
```

## 列表特性

比较大小的时候，不管长度如何，依次比较到第一个元素不相等的位置

比如[1, 2, 3] < [2, 3] 因为在比较1 < 2的时候就终止。

## 嵌套列表推导：展平二维数组

```python
nums = [e for row in matrix for e in row]
```

# Deque

```python
from collections import deque
list1 = [0, 1, 2, 3]
q=deque(list1)
q.append(4)    # 向右侧加	
q.appendleft(-1)    #向左侧加
q.extend(可迭代元素)    #向右侧添加可迭代元素
q.extendleft(可迭代元素)    
q=q.pop()    #移除最右端并返回元素值
l=q.popleft()    #移除最左端
q.count(1)    #统计元素个数    1
```

```python
# 返回string指定范围中str首次出现的位置
string.index(str, beg=0, end=len(string))
string.index(" ")
list(map(s.index,s))	#返回字符索引数组，如"abcba"->[0,1,2,1,0]
```

# 字典

```python
d.pop(key)	#返回key对应的value值，并在字典中删除这个键值对
d.get(key,default_value)	#获取key对应的值，如果不存在返回default_value
d.keys() 	#键构成的可迭代对象
d.values()	#值构成的可迭代对象
d.items()	#键值对构成的可迭代对象
d = defaultdict(list)	# 指定了具有默认值空列表的字典
```

# 字符串

## 1.字符串排序

```python
sorted(str) #返回按照字典序排序后的列表，如"eda"->['a','d','e']
s_sorted=''.join(sorted(str))	#把字符串列表组合成一个完整的字符串
```

## 2.最小表示法



# 区间合并

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = []
        l, r = intervals[0][0], intervals[0][1]
        for interval in intervals:
            il, ir = interval[0], interval[1]
            if il > r:
                res.append([l, r])
                l = il
            r = max(r, ir)
        res.append([l, r])
        return res
```

# 离散化

```python
tmp = nums.copy()
tmp.sort()
for i in range(n):
    nums[i] = bisect_sect(nums, tmp[i])

```

```python
tmp = sorted(nums)
mp = {}
for i, x in enumerate(tmp):
    mp[x] = i
```



# 二分查找

```python
from bisect import *
l = [1,1,1,3,3,3,4,4,4,5,5,5,8,9,10,10]
print(len(l)) # 16

print(bisect(l, 10))     # 相当于upper_bound, 16
print(bisect_right(l, 10))    

print(bisect_left(l, 10)) # 14
```

## 多维二分

```python
a = [(1, 20), (2, 19), (4, 15), (7,12)]
idx = bisect_left(a, (2, ))
```

# 前缀异或

```python
pre = list(accumulate(nums, xor, initial = 0))
```

# 优先队列



```python
from heapq import heapify, heappop, heappush
    heapify(nums)
    score = heappop(nums)
    heappush(nums, val)
# 注意：
# python中堆默认且只能是小顶堆
```

```python
nums = []
heapq.heappush(nums, val)	#插入
heapq.heappop(nums)			#弹出顶部
```



## 自定义比较规则

```python
class node():
    def __init__(self, need, get, idx):
        self.need = need
        self.get = get
        self.idx = idx
    def __lt__(self, other):
        return self.need < other.need
```

# 单调结构

## 单调栈

```python
    def trap(self, height: List[int]) -> int:
        # 单调栈：递减栈
        stk, n, res = deque(), len(height), 0
        for i in range(n):
            # 1.单调栈不为空、且违反单调性
            while stk and height[i] > height[stk[-1]]:
                # 2.出栈
                top = stk.pop()
                # 3.特判
                if not stk:
                    break
                # 4.获得左边界、宽度
                left = stk[-1]
                width =  i - left - 1
                # 5.计算
                res += (min(height[left], height[i]) - height[top]) *  width 
            # 6.入栈
            stk.append(i)
        return res
```

## 单调队列

```python
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = []
        q = deque()
        for i, x in enumerate(nums):
            # 1.入，需要维护单调减队列的有序性
            while q and x >= nums[q[-1]]:
                q.pop()
            q.append(i)

            # 2.出，当滑动窗口区间长度大于k的时候，弹出去左端的
            if i - q[0] + 1 > k:
                q.popleft()
            
            # 记录元素
            if i >= k - 1:
                res.append(nums[q[0]])
        return res    
```

# 前缀/差分

## 二维差分

```python
d = [[0] * (n + 2) for _ in range(m + 2)]
# 对矩阵中执行操作，使得左上角为(i, j)，右下角为(x, y)的矩阵都加k，等价于如下操作
d[i + 1][j + 1] += k
d[x + 2][y + 2] += k
d[i + 1][y + 2] -= k
d[x + 2][j + 1] -= k

# 还原差分时，直接原地还原
for i in range(m):
    for j in rang(n):
        d[i + 1][j + 1] += d[i][j + 1] + d[i + 1][j] - d[i][j]

```

# 数学

## 1.取整函数性质

### (1). 上下取整转换

$$
\left\lceil \frac{n}{m} \right\rceil = \left\lfloor \frac{n - 1}{m}  \right\rfloor + 1 = \left\lfloor \frac{n + m -1}{m} \right\rfloor
$$

### (2). 取余性质

$$
n \mod m = n - m \cdot \left\lfloor \frac{n}{m}\right\rfloor
$$

### (3). 幂等律

$$
\big\lfloor \left\lfloor x \right\rfloor \big\rfloor = \left\lfloor x \right\rfloor \\
\big\lceil \left\lceil x \right\rceil  \big\rceil = \left\lceil x \right\rceil
$$



## 2. 素数筛

埃氏筛：nloglogn

```python
is_prime = [True] * MX
is_prime[1] = is_prime[0] = False
for i in range(2, isqrt(MX) + 1):
    if is_prime[i]:
        for j in range(i * i, MX, i):
            is_prime[j] = False
```

## 3.其他简单数学
### 1.判断回文
```python
def is_par(x: int) -> bool:
	return x == int(str(x)[::-1])
```
### 2.pow函数
求$a^b \mod c$：

```python
pow(a, b, c)
```

### 3.求和公式

$$
\Sigma_1^nn^2 = \frac{n \cdot (n + 1) \cdot (2n + 1)}{6}
$$



## 4.数学公式

### 1.排序不等式

结论：$对于两个有序数组的乘积和，顺序和 \ge 乱序和 \ge 倒序和$。

$对于 a_1 \le a_2 \le \cdots \le a_n，b_1 \le b_2 \le \cdots \le b_n,并有c1,c2,\cdots, c_n是b1, b2, \cdots , b_n 的乱序排列。有如下关系： $
$$

\sum_{i = 1}^{n}a_ib_{n + 1 - i} \le \sum_{i=1}^{n}a_ic_i\le \sum_{i = 1}^{n}a_ib_i。\\
$$
$当且仅当 a_i = a_j 或者b_i = b_j \space (1 \le i, j\le n)时，等号成立。$

# 数据结构

## 字典树

```python
class Trie:

    def __init__(self):
        self.is_end = False
        self.next = [None] * 26

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            idx = ord(ch) - ord('a')
            if not node.next[idx]:
                node.next[idx] = Trie()
            node = node.next[idx]
        node.is_end = True            

    def search(self, word: str) -> bool:
        node = self
        for ch in word:
            idx = ord(ch) - ord('a')
            if not node.next[idx]:
                return False
            node = node.next[idx]
        return node.is_end    

    def startsWith(self, prefix: str) -> bool:
        node = self
        for ch in prefix:
            idx = ord(ch) - ord('a')
            if not node.next[idx]:
                return False
            node = node.next[idx]
        return True

```



## 动态开点 + lazy 线段树			

```python
# https://leetcode.cn/problems/range-module/
class Node:
    __slots__ = ['l', 'r', 'lazy', 'val']
    def __init__(self):
        self.l = None
        self.r = None
        self.lazy = 0
        self.val = False
class SegmentTree:
    __slots__ = ['root']
    def __init__(self):
        self.root = Node()


    def do(self, node, val):
        node.val = val
        node.lazy = 1

    # 下放lazy标记。如果是孩子为空，则动态开点
    def pushdown(self, node):
        if node.l is None:
            node.l = Node()
        if node.r is None:
            node.r = Node()
        
        # 根据lazy标记信息，更新左右节点，然后将lazy信息清除
        if node.lazy:
            self.do(node.l, node.val)
            self.do(node.r, node.val)
            node.lazy = 0

    def query(self, L, R, node = None, l = 1, r = int(1e9)):
        
        # 查询默认从根节点开始
        if node is None:
            node = self.root
        
        if L <= l and r <= R:
            return node.val

        # 下放标记、根据标记信息更新左右节点，然后清除标记
        self.pushdown(node)

        mid = (l + r) >> 1
        
        vl = vr = True
        
        if L <= mid:
            vl = self.query(L, R, node.l, l, mid)
        if R > mid:
            vr = self.query(L, R, node.r, mid + 1, r)
        return vl and vr
    
    
    def update(self, L, R, val, node = None, l = 1, r = int(1e9)):
        
        # 查询默认从根节点开始
        if node is None:
            node = self.root

        if L <= l and r <= R:
            self.do(node, val)
            return 

        mid = (l + r) >> 1

         # 下放标记、根据标记信息更新左右节点，然后清除标记
        self.pushdown(node)

        if L <= mid:
            self.update(L, R, val, node.l, l, mid)
        if R > mid:
            self.update(L, R, val, node.r, mid + 1, r)

        # node.val 为 True 表示这个节点所在区间，均被“跟踪”
        node.val = bool(node.l and node.l.val and node.r and node.r.val)


class RangeModule:

    def __init__(self):
        self.tree = SegmentTree()

    def addRange(self, left: int, right: int) -> None:
        self.tree.update(left, right - 1, True)

    def queryRange(self, left: int, right: int) -> bool:
        return self.tree.query(left, right - 1)

    def removeRange(self, left: int, right: int) -> None:
        self.tree.update(left, right - 1, False)


# Your RangeModule object will be instantiated and called as such:
# obj = RangeModule()
# obj.addRange(left,right)
# param_2 = obj.queryRange(left,right)
# obj.removeRange(left,right)
```

## 递归动态开点（无lazy) 线段树

区间覆盖统计问题，区间覆盖不需要重复操作，不需要进行lazy传递

但是数据范围较大，需要动态开点

```python
# https://leetcode.cn/problems/count-integers-in-intervals
class CountIntervals:
    __slots__ = 'left', 'right', 'l', 'r', 'val'

    def __init__(self, l = 1, r = int(1e9)):
        self.left = self.right = None
        self.l, self.r, self.val = l, r, 0


    def add(self, l: int, r: int) -> None:

        # 覆盖区间操作，不需要重复覆盖，饱和区间无需任何操作
        if self.val == self.r - self.l + 1: 
            return  

        if l <= self.l and self.r <= r:  # self 已被区间 [l,r] 完整覆盖，不再继续递归
            self.val = self.r - self.l + 1
            return
        
        
        mid = (self.l + self.r) >> 1
        
        # 动态开点
        if self.left is None: 
            self.left = CountIntervals(self.l, mid)  # 动态开点
        
        if self.right is None: 
            self.right = CountIntervals(mid + 1, self.r)  # 动态开点
        
        if l <= mid: 
            self.left.add(l, r)
        if mid < r: 
            self.right.add(l, r)
        
        # self.val 的值，表示区间[self.l, self.r] 中被覆盖的点的个数
        self.val = self.left.val + self.right.val

    def count(self) -> int:
        return self.val

```



## lazy线段树（内部）

```python
n = 0
tree, lazy = [], []
Nums = []

def build(i, l, r):
    if l == r:
        tree[i] = Nums[l - 1]
        return 
    mid = (l + r) >> 1
    build(i * 2, l, mid)
    build(i * 2 + 1, mid + 1, r)
    tree[i] = tree[i * 2] + tree[i * 2 + 1]

# 节点区间赋值、打上lazy标记
def do(i, l, r, val):
    tree[i] = (l - r + 1) * val
    lazy[i] = val

# 根据标记信息，更新子节点，设置子节点标记，清空标记
def pushdown(i, l, r):
    if lazy[i]:
        val = lazy[i]
        mid = (l + r) >> 1
        do(i * 2, l, mid, val)
        do(i * 2 + 1, mid + 1, r, val)
        lazy[i] = val

def Update(L, R, val, i = 1, l = 1, r = n):
    if L <= l and r <= R:
        do(i, l, r, val)
        return
    
    # 检查标记
    pushdown(i, l, r)
    mid = (l + r) >> 1
    if L <= mid:
        Update(L, R, val, i * 2, l, mid)
    if R > mid:
        Update(L, R, val, i * 2 + 1, mid + 1, r)
    
    # 更新节点区间
    tree[i] = tree[i * 2] + tree[i * 2 + 1]


def Query(L, R, i, l, r) -> int:
    if L <= l and r <= R:
        return tree[i]
    
    pushdown(i, l, r)
    mid = (l + r) >> 1
    vl = vr = 0
    if L <= mid:
        vl = Query(L, R, i * 2, l, mid)
    if R > mid:
        vr = Query(L, R, i * 2 + 1, mid + 1, r)
    return vl + vr


class NumArray:
    def __init__(self, nums: List[int]):
        global n, tree, lazy, Nums
        n = len(nums)
        tree = [0] * (4 * n)
        lazy = [0] * (4 * n)
        Nums = nums
        build(1, 1, n)

    def update(self, index: int, val: int) -> None:
        Update(index + 1, index + 1, val, 1, 1, n)

    def sumRange(self, left: int, right: int) -> int:
        return Query(left + 1, right + 1, 1, 1, n)


```



## lazy线段树（点区间赋值）

```python
class SegmentTree:
    __slots__ = ['node', 'lazy']
    def __init__(self, n: int):
        self.node = [0] * (4 * n)
        self.lazy = [0] * (4 * n)
    
    def build(self, i, l, r):
        if l == r:
            self.node[i] = Nums[l - 1]
            return
        mid = (l + r) >> 1
        
        self.build(i * 2, l ,mid)
        self.build(i * 2 + 1, mid + 1, r)

        self.node[i] = self.node[i * 2] + self.node[i * 2 + 1]

    # 更新节点值，设置lazy标记
    def do(self, i, l, r, val):
        self.node[i] = val * (l - r + 1)
        self.lazy[i] = val

    # 检查标记，根据标记根据子节点信息，下放标记，清除标记
    def pushdown(self, i, l, r):
        if self.lazy[i]:
            val = self.lazy[i]
            mid = (l + r) >> 1
            self.do(i * 2, l, mid, val)
            self.do(i * 2 + 1, mid + 1, r, val)
            self.lazy[i] = 0

    
    def update(self, i, l, r, L, R, val):
        if L <= l and r <= R:
            # 区间更新
            self.do(i, l, r, val)
            return 
        
        # 检查lazy标记
        self.pushdown(i, l, r)
        
        # 左右递归更新
        mid = (l + r) >> 1
        if L <= mid:
            self.update(i * 2, l, mid, L, R, val)
        if R > mid:
            self.update(i * 2 + 1, mid + 1, r, L, R, val)
        
        # 更新节点值: 区间和
        self.node[i] = self.node[i * 2] + self.node[i * 2 + 1]
    
    def query(self, i, l, r, L, R) -> int:
        if L <= l and r <= R:
            return self.node[i]
        
        # 检查lazy标记
        self.pushdown(i, l, r)

        mid = (l + r) >> 1

        vl, vr = 0, 0
        if L <= mid:
            vl = self.query(i * 2, l, mid, L, R)
        if R > mid:
            vr = self.query(i * 2 + 1, mid + 1, r, L, R)
        
        return vl + vr
```



## lazy 线段树（01翻转）

```python
class Solution:
    def handleQuery(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        node = [0] * (4 * n)
         # 懒标记：True表示该节点代表的区间被曾经被修改，但是其子节点尚未更新
        lazy = [False] * (4 * n)

        # 初始化线段树
        def build(i = 1, l = 1, r = n):
            if l == r:
                node[i] = nums1[l - 1]
                return 
            mid = (l + r) >> 1
            build(i * 2, l, mid)
            build(i * 2 + 1, mid + 1, r)
             # 维护区间 [l, r] 的值
            node[i] = node[i * 2] + node[i * 2 + 1]
        
        
        # 更新节点值，并设置lazy标记
        def do(i, l, r):
            node[i] = r - l + 1 - node[i]
            lazy[i] = not lazy[i]
        

        # 区间更新：本题中更新区间[l, r] 相当于做翻转
        def update(L, R, i = 1, l = 1, r = n):
            if L <= l and r <= R:
                do(i, l, r)
                return 
            
            mid = (l + r) >> 1
            if lazy[i]:
                # 根据标记信息更新p的两个左右子节点，同时为子节点增加标记
                # 然后清除当前节点的标记
                do(i * 2, l, mid)
                do(i * 2 + 1, mid + 1, r)
                lazy[i] = False
        
            if L <= mid:
                update(L, R, i * 2, l, mid)
            if R > mid:
                update(L, R, i * 2 + 1, mid + 1, r)
            
            # 更新节点值
            node[i] = node[i * 2] + node[i * 2 + 1]
        
        build()

        res, s = [], sum(nums2)
        for op, L, R in queries:
            if op == 1:
                update(L + 1, R + 1)
            elif op == 2:
                s += node[1] * L 
            else:
                res.append(s)
        return res
```

## 树状数组

```python
# 下标从1开始
class FenwickTree:
    def __init__(self, length: int):
        self.length = length
        self.tree = [0] * (length + 1)
    def lowbit(self, x: int) -> int:
        return x & (-x)

    # 更新自底向上
    def update(self, idx: int, val: int) -> None:
        while idx <= self.length:
            self.tree[idx] += val
            idx += self.lowbit(idx)

    # 查询自顶向下
    def query(self, idx: int) -> int:
        res = 0
        while idx > 0:    
            res += self.tree[idx]
            idx -= self.lowbit(idx)
        return res

class NumArray:

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.nums = nums
        self.tree = FenwickTree(n)
        for i, x in enumerate(nums):
            self.tree.update(i + 1, x)


    def update(self, index: int, val: int) -> None:
        # 因为这里是更新为val, 所以节点增加的值应为val - self.nums[index]
        # 同时需要更新nums[idx]
        self.tree.update(index + 1, val - self.nums[index])
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        r = self.tree.query(right + 1)
        l = self.tree.query(left)
        return r - l


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)
```

# 图论

## Floyd

```python
    mp = [[inf] * n for _ in range(n)]
    for u, v, w in edges:
        mp[u][v] = mp[v][u] = w
        mp[u][u] = mp[v][v] = 0
    for k in range(n):
        for u in range(n):
            for v in range(n):
                mp[u][v] = min(mp[u][v], mp[u][k] + mp[k][v])
```

## 倍增LCA

$f[u][i] 表示 u 节点 向上跳2^i\space 的节点$，$dep[u] \space 表示深度$

```python
    MX = int(n.bit_length())
    f = [[0] * (MX + 1) for _ in range(n)]
    dep = [0] * n

    def dfs(u, fa):
        # father[u] = fa
        dep[u] = dep[fa] + 1    # 递归节点深度
        f[u][0] = fa
        for i in range(1, MX + 1):  # 倍增计算向上跳的位置
            f[u][i] = f[f[u][i - 1]][i - 1]
        for v in g[u]:
            if v != fa:
                dfs(v, u)

    # 假定0节点是树根
    dep[0] = 1
    for v in g[0]:
        dfs(v, 0)

    def lca(u, v):
        if dep[u] < dep[v]:
            u, v = v, u
        # u 跳到和v 同一层
        for i in range(MX, -1, -1):
            if dep[f[u][i]] >= dep[v]:
                u = f[u][i]
        if u == v:
            return u
        # 跳到lca的下一层
        for i in range(MX, -1, -1):
            if f[u][i] != f[v][i]:
                u, v = f[u][i], f[v][i]
        return f[u][0]
```

## 树上差分

点差分：解决多路径节点计数问题。

$u \rightarrow v 的路径转化为 u \rightarrow lca左孩子 + lca \rightarrow v$ 

```python
# 差分时左闭右开，无需考虑啊u = a的情况
for u, v in query:
    a = lca(u, v)
    diff[u] += 1
    diff[a] -= 1
    diff[v] += 1
    if father[a] != -1:
        diff[father[a]] -= 1
```

![image.png](https://pic.leetcode.cn/1701864888-CJnrkG-image.png)

## 并查集

`find(u) == find(v)` 表示u, v在同一集合

```python
    fa = list(range(n + 1))

    # 查找x集合的根
    def find(x):
        if fa[x] != x:
            fa[x] = find(fa[x])
        return fa[x]

    # v并向u中
    def union(u, v):
        fa[find(v)] = find(u)
```

# 位运算/状态压缩

## 1.二维矩阵 压缩为一维二进制串

```python
num = sum((ch == '.') << i for i, ch in enumerate(s))	# 010110
```

满足 $num >> x == s[i]$

```python
s = ["#", ".", ".", "#", ".", "#"]
num = sum((ch == '.') << i for i, ch in enumerate(s))	# 010110
print(bin(num))	# 0b 010110
```

## 2.枚举一个二进制串的子集

```python
s = 19
j = s
while j:
    # print(format(j, '06b'))
    j = (j - 1) & s
```

## 3.判断是否有两个连续（相邻）的1

```python
(s & (s >> 1)) == 0	# 为True是表示没有两个连续的1
或者
(s & (s << 1)) == 0 
```

## 4.二进制

#### 十进制长度

```python
m = int(log(n + 1, 10)) + 1
```

#### 二进制长度	

```python
n = num.bit_lenght()
```

#### 二进制中1的数量

```python
cnt = num.bit_count()
```

## 5.最大异或

```python
def findMaximumXOR(self, nums: List[int]) -> int:
        n = max(nums).bit_length()
        res = mask = 0
        for i in range(n - 1, -1, -1):
            mask |= 1 << i 
            s, tmp = set(), res | (1 << i)
            for x in nums: # x ^ a = tmp -> a = tmp ^ x
                x &= mask
                if tmp ^ x in s:
                    res = tmp
                    break
                s.add(x)
        return res
```

# 数位dp
```python
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        low = str(start)
        high = str(finish)
        n = len(high)
        low = '0' * (n - len(low)) + low # 补全前导0
        diff = n - len(s)

        @lru_cache(maxsize = None)
        def dfs(i, limit_low: bool, limit_high: bool) -> int:
            if i == n:
                return 1
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            res = 0
            if i < diff:    # 枚举这个位填什么
                for d in range(lo, min(hi, limit) + 1):
                    res += dfs(i + 1, limit_low and d == lo, limit_high and d == hi)
            else:
                x = int(s[i - diff])
                if lo <= x <= min(hi, limit):
                    res = dfs(i + 1, limit_low and x == lo, limit_high and x == high)
            return res
        return dfs(0, True, True)
```
