from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        s = sum(nums)
        if s < target: return 0
        n = len(nums)
        f = [0] * (2 * s + 1)
        p = [0] * (2 * s + 1)
        p[s] = 1
        for x in nums:
            print(p)
            for j in range(2 * s + 1):
                if j - x >= 0:
                    f[j] += p[j - x]
                if j + x < 2 * s + 1:
                    f[j] += p[j + x]
            for i in range(2 * s + 1):
                p[i] = f[i]
        return f[target + s]

class Solution2:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        
        # f[i][j] 从前i 个中，在偏移量为s情况下，
        # 目标和等于 j 的所有选法的集合的方案数
        # f[n][target + s]
        # f[0][s] = 1
        # 最多 f[n][2 * s]
        s = sum(nums)
        if s < target: return 0
        n = len(nums)
        f = [[0] * (2 * s + 1) for _ in range(n + 1)]
        f[0][s] = 1
        for i in range(1, n + 1):
            print(f[i - 1])
            for j in range(2 * s + 1):
                if j - nums[i - 1] >= 0:
                    f[i][j] += f[i - 1][j - nums[i - 1]]
                if j + nums[i - 1] < 2 * s + 1:
                    f[i][j] += f[i - 1][j + nums[i - 1]]
        return f[n][target + s]
s = Solution()
print(s.findTargetSumWays([1, 1, 1, 1, 1], 3))  # 5

s = Solution2()
print(s.findTargetSumWays([1, 1, 1, 1, 1], 3))  # 5