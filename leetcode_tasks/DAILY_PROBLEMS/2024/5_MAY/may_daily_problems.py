# 1 MAY

# https://leetcode.com/problems/reverse-prefix-of-word/description/?envType=daily-question&envId=2024-05-01

# class Solution:
#     def reversePrefix(self, word: str, ch: str) -> str:
#         j = word.find(ch)
#         if j != -1:
#             return word[:j+1][::-1] + word[j+1:]
#         return word