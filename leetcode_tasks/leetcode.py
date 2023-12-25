#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/longest-substring-without-repeating-characters/description/ |
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

# #~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         set_s = set(s)
#         if "p" in set_s:
#             set_s.remove("p" or "P")
#         len_s = len(set_s)
#         return len_s

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         n = len(s)
#         char_index_map = {}
#         max_length = 0
#         start = 0
#
#         for end in range(n):
#             if s[end] in char_index_map and char_index_map[s[end]] >= start:
#                 start = char_index_map[s[end]] + 1
#
#             char_index_map[s[end]] = end
#             max_length = max(max_length, end - start + 1)
#
#         return max_length
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~