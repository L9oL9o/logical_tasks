# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/largest-substring-between-two-equal-characters/?envType=daily-question&envId=2023-12-31 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def maxLengthBetweenEqualCharacters(self, s: str) -> int:
#         first_occurrence = {}
#         max_length = -1
#
#         for i, char in enumerate(s):
#             if char in first_occurrence:
#                 max_length = max(max_length, i - first_occurrence[char] - 1)
#             else:
#                 first_occurrence[char] = i
#
#         return max_length
