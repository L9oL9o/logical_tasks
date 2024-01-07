# 7 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/length-of-last-word |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def lengthOfLastWord(self, s: str) -> int:
#         words = s.split()
#         if words:
#             s = words[-1]
#             return len(s)

# ~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def lengthOfLastWord(self, s: str) -> int:
#         return len(s.strip().split(' ')[-1])