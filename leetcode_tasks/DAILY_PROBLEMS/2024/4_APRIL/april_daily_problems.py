# 01 APRIL
# https://leetcode.com/problems/length-of-last-word/description/?envType=daily-question&envId=2024-04-01

# class Solution:
#     def lengthOfLastWord(self, s: str) -> int:
#         words = s.strip().split()
#
#         if not words:
#             return 0
#
#         return len(words[-1])


# 02 APRIL
# https://leetcode.com/problems/isomorphic-strings/description/?envType=daily-question&envId=2024-04-02

# class Solution:
#     def isIsomorphic(self, s: str, t: str) -> bool:
#         indexS = [0] * 200  # Stores index of characters in string s
#         indexT = [0] * 200  # Stores index of characters in string t
#
#         length = len(s)  # Get the length of both strings
#
#         if length != len(t):  # If the lengths of the two strings are different, they can't be isomorphic
#             return False
#
#         for i in range(length):  # Iterate through each character of the strings
#             if indexS[ord(s[i])] != indexT[ord(t[
#                                                    i])]:  # Check if the index of the current character in string s is different from the index of the corresponding character in string t
#                 return False  # If different, strings are not isomorphic
#
#             indexS[ord(s[i])] = i + 1  # updating position of current character
#             indexT[ord(t[i])] = i + 1
#
#         return True  # If the loop completes without returning false, strings are isomorphic


# 03 APRIL
# https://leetcode.com/problems/word-search/description/?envType=daily-question&envId=2024-04-03

# class Solution:
#     def exist(self, board, word):
#         def backtrack(i, j, k):
#             if k == len(word):
#                 return True
#             if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#                 return False
#
#             temp = board[i][j]
#             board[i][j] = ''
#
#             if backtrack(i + 1, j, k + 1) or backtrack(i - 1, j, k + 1) or backtrack(i, j + 1, k + 1) or backtrack(i,
#                                                                                                                    j - 1,
#                                                                                                                    k + 1):
#                 return True
#
#             board[i][j] = temp
#             return False
#
#         for i in range(len(board)):
#             for j in range(len(board[0])):
#                 if backtrack(i, j, 0):
#                     return True
#         return False

# 04 APRIL
# https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/description/?envType=daily-question&envId=2024-04-04

# class Solution:
#     def maxDepth(self, s):
#         count = 0
#         max_num = 0
#         for i in s:
#             if i == "(":
#                 count += 1
#                 if max_num < count:
#                     max_num = count
#             if i == ")":
#                 count -= 1
#         return(max_num)


# 05 APRIL
# https://leetcode.com/problems/make-the-string-great/description/?envType=daily-question&envId=2024-04-05

# class Solution:
#     def makeGood(self, s: str) -> str:
#         stack = []
#         for char in s:
#             if stack and abs(ord(char) - ord(stack[-1])) == 32:
#                 stack.pop()
#             else:
#                 stack.append(char)
#
#         return ''.join(stack)


# 06 APRIL
# https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/description/?envType=daily-question&envId=2024-04-06

# class Solution:
#     def makeGood(self, s: str) -> str:
#         stack = []
#         for char in s:
#             if stack and abs(ord(char) - ord(stack[-1])) == 32:
#                 stack.pop()
#             else:
#                 stack.append(char)
#
#         return ''.join(stack)


# 07 APRIL
# https://leetcode.com/problems/valid-parenthesis-string/description/?envType=daily-question&envId=2024-04-07

# class Solution:
#     def checkValidString(self, s: str) -> bool:
#         leftMin, leftMax = 0, 0
#
#         for c in s:
#             if c == "(":
#                 leftMin, leftMax = leftMin + 1, leftMax + 1
#             elif c == ")":
#                 leftMin, leftMax = leftMin - 1, leftMax - 1
#             else:
#                 leftMin, leftMax = leftMin - 1, leftMax + 1
#             if leftMax < 0:
#                 return False
#             if leftMin < 0:
#                 leftMin = 0
#         return leftMin == 0


# 08 APRIL
#

# 09 APRIL
#

# 10 APRIL
#

# 11 APRIL
#

# 12 APRIL
#

# 13 APRIL
#

# 14 APRIL
#

# 15 APRIL
#

# 16 APRIL
#

# 17 APRIL
#

# 18 APRIL
#

# 19 APRIL
#

# 20 APRIL
#

# 21 APRIL
#

# 22 APRIL
#

# 23 APRIL
#

# 24 APRIL
#

# 25 APRIL
#

# 26 APRIL
#

# 27 APRIL
#

# 28 APRIL
#

# 29 APRIL
#

# 30 APRIL
#

