# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!~~~~~|
# # https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/?envType=daily-question&envId=2023-12-29 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
#         n = len(jobDifficulty)
#         if n < d:
#             return -1
#
#         # dp[i][j]: minimum difficulty to schedule i jobs in j days
#         dp = [[float('inf')] * (d + 1) for _ in range(n + 1)]
#         dp[0][0] = 0
#
#         for i in range(1, n + 1):
#             for k in range(1, d + 1):
#                 max_difficulty = 0
#                 for j in range(i - 1, -1, -1):
#                     max_difficulty = max(max_difficulty, jobDifficulty[j])
#                     dp[i][k] = min(dp[i][k], dp[j][k - 1] + max_difficulty)
#
#         return dp[n][d] if dp[n][d] != float('inf') else -1
