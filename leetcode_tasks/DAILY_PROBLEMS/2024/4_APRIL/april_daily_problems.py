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
# https://leetcode.com/problems/island-perimeter/description/?envType=daily-question&envId=2024-04-18

# class Solution:
#     def islandPerimeter(self, grid: List[List[int]]) -> int:
#         perimeter = 0
#         rows, cols = len(grid), len(grid[0])
#
#         for r in range(rows):
#             for c in range(cols):
#                 if grid[r][c] == 1:
#                     perimeter += 4
#                     if r > 0 and grid[r - 1][c] == 1:
#                         perimeter -= 2
#                     if c > 0 and grid[r][c - 1] == 1:
#                         perimeter -= 2
#
#         return perimeter


# 19 APRIL
# https://leetcode.com/problems/number-of-islands/description/?envType=daily-question&envId=2024-04-19

# class Solution:
#     def numIslands(self, grid: List[List[str]]) -> int:
#         if not grid:
#             return 0
#
#         def dfs(i, j):
#             if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
#                 return
#             grid[i][j] = '0'  # mark as visited
#             dfs(i + 1, j)
#             dfs(i - 1, j)
#             dfs(i, j + 1)
#             dfs(i, j - 1)
#
#         num_islands = 0
#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if grid[i][j] == '1':
#                     num_islands += 1
#                     dfs(i, j)
#
#         return num_islands


# 20 APRIL
# https://leetcode.com/problems/find-all-groups-of-farmland/description/?envType=daily-question&envId=2024-04-20

# class Solution:
#     def findFarmland(self, land: List[List[int]]) -> List[List[int]]:
#         def dfs(x, y):
#             # This function performs DFS to mark all connected farmland and find the boundaries
#             stack = [(x, y)]
#             min_row, min_col = x, y
#             max_row, max_col = x, y
#             visited.add((x, y))
#
#             while stack:
#                 cur_x, cur_y = stack.pop()
#                 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                     nx, ny = cur_x + dx, cur_y + dy
#                     if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and land[nx][ny] == 1:
#                         visited.add((nx, ny))
#                         stack.append((nx, ny))
#                         min_row = min(min_row, nx)
#                         min_col = min(min_col, ny)
#                         max_row = max(max_row, nx)
#                         max_col = max(max_col, ny)
#
#             return (min_row, min_col, max_row, max_col)
#
#         rows, cols = len(land), len(land[0])
#         visited = set()
#         result = []
#
#         for i in range(rows):
#             for j in range(cols):
#                 if land[i][j] == 1 and (i, j) not in visited:
#                     # Found a new piece of farmland
#                     min_row, min_col, max_row, max_col = dfs(i, j)
#                     result.append([min_row, min_col, max_row, max_col])
#
#         return result


# 21 APRIL
# https://leetcode.com/problems/find-if-path-exists-in-graph/description/?envType=daily-question&envId=2024-04-21

# class Solution:
#     def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
#         graph = collections.defaultdict(list)
#         for u, v in edges:
#             graph[u].append(v)
#             graph[v].append(u)
#
#         def dfs(node, visited):
#             if node == destination:
#                 return True
#             visited.add(node)
#             for neighbor in graph[node]:
#                 if neighbor not in visited:
#                     if dfs(neighbor, visited):
#                         return True
#             return False
#
#         visited = set()
#         return dfs(source, visited)


# 22 APRIL
# https://leetcode.com/problems/open-the-lock/description/?envType=daily-question&envId=2024-04-22

# class Solution:
#     def openLock(self, deadends: List[str], target: str) -> int:
#         # Convert deadends to a set for O(1) lookup
#         deadends = set(deadends)
#         if "0000" in deadends:
#             return -1
#
#         # Initialize BFS
#         queue = deque([('0000', 0)])  # (current_combination, moves)
#         visited = set('0000')
#
#         # BFS loop
#         while queue:
#             current_combination, moves = queue.popleft()
#
#             # Check if we've reached the target
#             if current_combination == target:
#                 return moves
#
#             # Generate next possible combinations
#             for i in range(4):
#                 for delta in [-1, 1]:
#                     new_digit = (int(current_combination[i]) + delta) % 10
#                     new_combination = (
#                             current_combination[:i] + str(new_digit) + current_combination[i + 1:]
#                     )
#
#                     # Check if the new combination is valid and not visited
#                     if new_combination not in visited and new_combination not in deadends:
#                         visited.add(new_combination)
#                         queue.append((new_combination, moves + 1))
#
#         # If target is not reachable
#         return -1


# 23 APRIL
# https://leetcode.com/problems/minimum-height-trees/description/?envType=daily-question&envId=2024-04-23

# class Solution:
#     def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
#         if n == 1:
#             return [0]
#
#         # Initialize the adjacency list and degree of each node
#         adjacency_list = defaultdict(list)
#         degree = [0] * n
#         for u, v in edges:
#             adjacency_list[u].append(v)
#             adjacency_list[v].append(u)
#             degree[u] += 1
#             degree[v] += 1
#
#         # Initialize leaves queue
#         leaves = deque([i for i in range(n) if degree[i] == 1])
#
#         # Trim leaves until 2 or fewer nodes remain
#         remaining_nodes = n
#         while remaining_nodes > 2:
#             leaves_count = len(leaves)
#             remaining_nodes -= leaves_count
#             for _ in range(leaves_count):
#                 leaf = leaves.popleft()
#                 for neighbor in adjacency_list[leaf]:
#                     degree[neighbor] -= 1
#                     if degree[neighbor] == 1:
#                         leaves.append(neighbor)
#
#         return list(leaves)


# 24 APRIL
# https://leetcode.com/problems/n-th-tribonacci-number/description/?envType=daily-question&envId=2024-04-24

# class Solution:
#     def tribonacci(self, n: int) -> int:
#         dp=[-1 for _ in range(n+1)]
#
#         def dfs(n):
#             if n==2 or n==1:
#                 return 1
#             if n==0:
#                 return 0
#             if dp[n] != -1:
#                 return dp[n]
#             dp[n]=dfs(n-2)+dfs(n-1)+dfs(n-3)
#             return dp[n]
#
#         return dfs(n)


# 25 APRIL
# https://leetcode.com/problems/longest-ideal-subsequence/?envType=daily-question&envId=2024-04-25

# class Solution:
#     def longestIdealString(self, s: str, k: int) -> int:
#         dp = [0] * 27
#         n = len(s)
#
#         for i in range(n - 1, -1, -1):
#             cc = s[i]
#             idx = ord(cc) - ord('a')
#             maxi = float('-inf')
#
#             left = max((idx - k), 0)
#             right = min((idx + k), 26)
#
#             for j in range(left, right + 1):
#                 maxi = max(maxi, dp[j])
#
#             dp[idx] = maxi + 1
#
#         return max(dp)


# 26 APRIL
# https://leetcode.com/problems/minimum-falling-path-sum-ii/description/?envType=daily-question&envId=2024-04-26

# class Solution:
#     def minFallingPathSum(self, grid: List[List[int]]) -> int:
#         n, res = len(grid), float('inf')
#         dp = [[-1] * n for _ in range(n)]
#
#         for j in range(n):
#             dp[0][j] = grid[0][j]
#
#         for i in range(1, n):
#             for j in range(n):
#                 temp = float('inf')
#
#                 for k in range(n):
#                     if j != k:
#                         temp = min(temp, grid[i][j] + dp[i - 1][k])
#
#                 dp[i][j] = temp
#
#         for j in range(n):
#             res = min(res, dp[n - 1][j])
#
#         return res


# 27 APRIL
# https://leetcode.com/problems/freedom-trail/description/?envType=daily-question&envId=2024-04-27

# from collections import defaultdict
#
#
# class Solution:
#     def findRotateSteps(self, ring: str, key: str) -> int:
#         memo = {}
#         positions = defaultdict(list)
#         for i, c in enumerate(ring):
#             positions[c].append(i)
#         return self.helper(0, 0, positions, key, ring, memo)
#
#     def helper(self, in_index, pos, positions, key, ring, memo):
#         if in_index == len(key):
#             return 0
#         if (in_index, pos) in memo:
#             return memo[(in_index, pos)]
#         min_steps = float('inf')
#         for i in positions[key[in_index]]:
#             if i >= pos:
#                 steps = min(i - pos, pos + len(ring) - i)
#             else:
#                 steps = min(pos - i, i + len(ring) - pos)
#             min_steps = min(min_steps, steps + self.helper(in_index + 1, i, positions, key, ring, memo))
#         memo[(in_index, pos)] = min_steps + 1
#         return min_steps + 1


# 28 APRIL
# https://leetcode.com/problems/sum-of-distances-in-tree/submissions/1244261248/?envType=daily-question&envId=2024-04-28

# class Solution:
#     def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
#         graph = defaultdict(list)
#         for u, v in edges:
#             graph[u].append(v)
#             graph[v].append(u)
#
#         count = [1] * n
#         res = [0] * n
#
#         def dfs(node, parent):
#             for child in graph[node]:
#                 if child != parent:
#                     dfs(child, node)
#                     count[node] += count[child]
#                     res[node] += res[child] + count[child]
#
#         def dfs2(node, parent):
#             for child in graph[node]:
#                 if child != parent:
#                     res[child] = res[node] - count[child] + (n - count[child])
#                     dfs2(child, node)
#
#         dfs(0, -1)
#         dfs2(0, -1)
#
#         return res


# 29 APRIL
# https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/?envType=daily-question&envId=2024-04-29

# class Solution:
#     def minOperations(self, nums: List[int], k: int) -> int:
#         final_xor = 0
#         # XOR of all integers in the array.
#         for n in nums:
#             final_xor = final_xor ^ n
#
#         count = 0
#         # Keep iterating until both k and final_xor becomes zero.
#         while k or final_xor:
#             # k % 2 returns the rightmost bit in k,
#             # final_xor % 2 returns the rightmost bit in final_xor.
#             # Increment counter, if the bits don't match.
#             if (k % 2) != (final_xor % 2):
#                 count += 1
#
#             # Remove the last bit from both integers.
#             k //= 2
#             final_xor //= 2
#
#         return count


# 30 APRIL
# https://leetcode.com/problems/number-of-wonderful-substrings/description/?envType=daily-question&envId=2024-04-30

# class Solution(object):
#     def wonderfulSubstrings(self, word):
#         count = [0] * 1024  # 2^10 to store XOR values
#         result = 0
#         prefix_xor = 0
#         count[prefix_xor] = 1
#
#         for char in word:
#             char_index = ord(char) - ord('a')
#             prefix_xor ^= 1 << char_index
#             result += count[prefix_xor]
#             for i in range(10):
#                 result += count[prefix_xor ^ (1 << i)]
#             count[prefix_xor] += 1
#
#         return result