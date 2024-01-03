# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/valid-parentheses/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def isValid(self, s: str) -> bool:
#         brackets1 = "()"
#         brackets2 = "[]"
#         brackets3 = "{}"
#         check_bracket = ""
#         for i in range(len(s)):
#             for j in range(len(s) + 1):
#                 if s[i] and s[j] == brackets1:
#                     check_bracket += s[i], s[j]
#                 elif s[i] and s[j] == brackets2:
#                     check_bracket += s[i], s[j]
#                 elif s[i] and s[j] == brackets3:
#                     check_bracket += s[i], s[j]
#         return check_bracket

# ~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def isValid(self, s: str) -> bool:
#         stack = []
#         bracket_pairs = {')': '(', '}': '{', ']': '['}
#         for char in s:
#             if char in bracket_pairs.values():
#                 stack.append(char)
#             elif char in bracket_pairs.keys():
#                 if not stack or bracket_pairs[char] != stack.pop():
#                     return False
#             else:
#                 # If the character is not an open or close bracket, ignore it
#                 continue
#         # The string is valid if the stack is empty after processing all characters
#         return not stack

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def isValid(self, s: str) -> bool:
#         check = {
#             '(': ')',
#             '{': '}',
#             '[': ']'
#         }
#         stack = []
#         for c in s:
#             if stack and stack[-1] in check and check[stack[-1]] == c:
#                 stack.pop()
#             else:
#                 stack.append(c)
#         return not stack