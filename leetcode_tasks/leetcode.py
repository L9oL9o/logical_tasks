# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/string-to-integer-atoi/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def myAtoi(self, s: str) -> int:
#         # Step 1: Read in and ignore any leading whitespace.
#         i = 0
#         while i < len(s) and s[i].isspace():
#             i += 1
#
#         # Step 2: Check if the next character is '-' or '+'
#         if i < len(s) and (s[i] == '-' or s[i] == '+'):
#             sign = -1 if s[i] == '-' else 1
#             i += 1
#         else:
#             sign = 1
#
#         # Step 3: Read in the characters until the next non-digit character or end of input
#         result = 0
#         while i < len(s) and s[i].isdigit():
#             digit = int(s[i])
#             # Step 4: Convert digits into an integer
#             result = result * 10 + digit
#             i += 1
#
#         # Step 5: Change the sign if necessary
#         result *= sign
#
#         # Step 6: Clamp the result to the 32-bit signed integer range
#         INT_MAX = 2**31 - 1
#         INT_MIN = -2**31
#         result = max(min(result, INT_MAX), INT_MIN)
#
#         return result

# ~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import re
#
# class Solution:
#     def myAtoi(self, s: str) -> int:
#         s = s.strip()
#         s = re.findall("^[+\-]?\d+",s)
#         if not s: return 0
#         n = int("".join(s))
#         MAX = pow(2,31)-1
#         MIN = -pow(2,31)
#         if n>MAX: return MAX
#         if n<MIN: return MIN
#         return n


