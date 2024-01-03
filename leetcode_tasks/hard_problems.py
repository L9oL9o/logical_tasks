# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/median-of-two-sorted-arrays/submissions/1129949277/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
#         if len(nums1) > len(nums2):
#             nums1, nums2 = nums2, nums1
#
#         x, y = len(nums1), len(nums2)
#         low, high = 0, x
#
#         while low <= high:
#             partitionX = (low + high) // 2
#             partitionY = (x + y + 1) // 2 - partitionX
#
#             maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
#             minX = float('inf') if partitionX == x else nums1[partitionX]
#
#             maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
#             minY = float('inf') if partitionY == y else nums2[partitionY]
#
#             if maxX <= minY and maxY <= minX:
#                 if (x + y) % 2 == 0:
#                     return (max(maxX, maxY) + min(minX, minY)) / 2.0
#                 else:
#                     return max(maxX, maxY)
#             elif maxX > minY:
#                 high = partitionX - 1
#             else:
#                 low = partitionX + 1
#



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



