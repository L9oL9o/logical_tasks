# ~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/regular-expression-matching/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         # Create a 2D DP array to store matching results
#         dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
#
#         # Empty string and empty pattern match
#         dp[0][0] = True
#
#         # Handle patterns with '*'
#         for j in range(1, len(p) + 1):
#             if p[j - 1] == '*':
#                 dp[0][j] = dp[0][j - 2]
#
#         # Fill in the DP array
#         for i in range(1, len(s) + 1):
#             for j in range(1, len(p) + 1):
#                 if p[j - 1] == s[i - 1] or p[j - 1] == '.':
#                     dp[i][j] = dp[i - 1][j - 1]
#                 elif p[j - 1] == '*':
#                     dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] if s[i - 1] == p[j - 2] or p[j - 2] == '.' else False)
#
#         return dp[-1][-1]