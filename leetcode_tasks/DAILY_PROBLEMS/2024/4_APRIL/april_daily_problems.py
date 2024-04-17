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
# https://leetcode.com/problems/number-of-students-unable-to-eat-lunch/description/?envType=daily-question&envId=2024-04-08

# class Solution:
#     def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
#         count = [0, 0]
#         for student in students:
#             count[student] += 1
#
#         for i in range(len(sandwiches)):
#             if count[sandwiches[i]] == 0:
#                 return len(sandwiches) - i
#             count[sandwiches[i]] -= 1
#
#         return 0


# 09 APRIL
# https://leetcode.com/problems/time-needed-to-buy-tickets/description/?envType=daily-question&envId=2024-04-09

# class Solution:
#     def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
#         total = 0
#
#         for i, x in enumerate(tickets):
#             if i <= k:
#                 total += min(tickets[i], tickets[k])
#             else:
#                 total += min(tickets[i], tickets[k] - 1)
#
#         return total


# 10 APRIL
# https://leetcode.com/problems/reveal-cards-in-increasing-order/description/?envType=daily-question&envId=2024-04-10

# class Solution:
#     def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
#         # Sort the deck in increasing order
#         deck.sort()
#
#         n = len(deck)
#         result = [0] * n
#         indices = deque(range(n))
#
#         for card in deck:
#             idx = indices.popleft()  # Get the next available index
#             result[idx] = card  # Place the card in the result array
#             if indices:  # If there are remaining indices in the deque
#                 indices.append(indices.popleft())  # Move the used index to the end of deque
#
#         return result


# 11 APRIL
# https://leetcode.com/problems/remove-k-digits/description/?envType=daily-question&envId=2024-04-11

# class Solution:
#     def removeKdigits(self, num: str, k: int) -> str:
#         stack = []
#         for c in num:
#             while k and stack and stack[-1] > c:
#                 stack.pop()
#                 k -= 1
#             stack.append(c)
#         if k:
#             stack = stack[:-k]
#         i = 0
#         while i < len(stack) and stack[i] == "0":
#             i += 1
#
#         ret = "".join(stack[i:])
#         return "0" if not ret else ret


# 12 APRIL
# https://leetcode.com/problems/trapping-rain-water/description/?envType=daily-question&envId=2024-04-12

# class Solution:
#     def trap(self, height: List[int]) -> int:
#         i = 0
#         left_max = height[0]
#         sum = 0
#         j = len(height) - 1
#         right_max = height[j]
#         while i < j:
#             if left_max <= right_max:
#                 sum += left_max - height[i]
#                 i += 1
#                 left_max = max(left_max, height[i])
#             else:
#                 sum += right_max - height[j]
#                 j -= 1
#                 right_max = max(right_max, height[j])
#         return sum


# 13 APRIL
# https://leetcode.com/problems/maximal-rectangle/description/?envType=daily-question&envId=2024-04-13

# class Solution:
#     def maximalRectangle(self, matrix: List[List[str]]) -> int:
#         r, c = len(matrix), len(matrix[0])
#         if r == 1 and c == 1:
#             if matrix[0][0] == '1':
#                 return 1
#             else:
#                 return 0
#         h = [0] * (c + 1)
#         maxArea = 0
#
#         for i, row in enumerate(matrix):
#             st = [-1]
#             row.append('0')
#             # build h
#             for j, x in enumerate(row):
#                 if x == '1':
#                     h[j] += 1
#                 else:
#                     h[j] = 0
#                 # mononotonic stack has at leat element -1
#                 while len(st) > 1 and (j == c or h[j] < h[st[-1]]):
#                     m = st[-1]
#                     st.pop()
#                     w = j - st[-1] - 1
#                     area = h[m] * w
#                     maxArea = max(maxArea, area)
#                 st.append(j)
#         return maxArea


# 14 APRIL
# https://leetcode.com/problems/sum-of-left-leaves/description/?envType=daily-question&envId=2024-04-14

# class Solution:
#     def sumOfLeftLeaves(self, root: TreeNode) -> int:
#         if not root:
#             return 0
#
#         ans = 0
#
#         if root.left:
#             if not root.left.left and not root.left.right:
#                 ans += root.left.val
#             else:
#                 ans += self.sumOfLeftLeaves(root.left)
#
#         ans += self.sumOfLeftLeaves(root.right)
#
#         return ans


# 15 APRIL
# https://leetcode.com/problems/sum-root-to-leaf-numbers/description/?envType=daily-question&envId=2024-04-15

# class Solution:
#     def sumNumbers(self, root: TreeNode) -> int:
#         def dfs(node, path):
#             nonlocal ans
#             if not node:
#                 return
#             if not node.left and not node.right:
#                 ans += path * 10 + node.val
#                 return
#             dfs(node.left, path * 10 + node.val)
#             dfs(node.right, path * 10 + node.val)
#
#         ans = 0
#         dfs(root, 0)
#         return ans


# 16 APRIL
# https://leetcode.com/problems/add-one-row-to-tree/description/?envType=daily-question&envId=2024-04-16

# class Solution:
#     def add(self, root, val, depth, curr):
#         if not root:
#             return None
#
#         if curr == depth - 1:
#             lTemp = root.left
#             rTemp = root.right
#
#             root.left = TreeNode(val)
#             root.right = TreeNode(val)
#             root.left.left = lTemp
#             root.right.right = rTemp
#
#             return root
#
#         root.left = self.add(root.left, val, depth, curr + 1)
#         root.right = self.add(root.right, val, depth, curr + 1)
#
#         return root
#
#     def addOneRow(self, root, val, depth):
#         if depth == 1:
#             newRoot = TreeNode(val)
#             newRoot.left = root
#             return newRoot
#
#         return self.add(root, val, depth, 1)


# 17 APRIL
# https://leetcode.com/problems/smallest-string-starting-from-leaf/description/?envType=daily-question&envId=2024-04-17

# class Solution:
#     def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
#         # Helper function to perform DFS
#         def dfs(node, path, smallest):
#             if not node:
#                 return
#
#             # Append current node's character to the path
#             path.append(chr(node.val + ord('a')))
#
#             # If it's a leaf node, reverse the path and compare
#             if not node.left and not node.right:
#                 current_string = ''.join(path[::-1])  # reverse path to get string
#                 smallest[0] = min(smallest[0], current_string)
#
#             # Recursively traverse left and right subtrees
#             dfs(node.left, path, smallest)
#             dfs(node.right, path, smallest)
#
#             # Backtrack: remove the current node's character from the path
#             path.pop()
#
#         # Initialize smallest string as a large value
#         smallest = [chr(ord('z') + 1)]  # Store smallest string found
#
#         # Start DFS from the root with an empty path
#         dfs(root, [], smallest)
#
#         return smallest[0]


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

