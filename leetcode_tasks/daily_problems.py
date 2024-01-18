# 21 DECEMBER 2023
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/widest-vertical-area-between-two-points-containing-no-points/description/?envType=daily-question&envId=2023-12-21 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
#         points.sort(key=lambda x: x[0])  # Sort points based on x-coordinate
#
#         max_width = 0
#         for i in range(1, len(points)):
#             max_width = max(max_width, points[i][0] - points[i - 1][0])
#
#         return max_width


# 1 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/assign-cookies/?envType=daily-question&envId=2024-01-01 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findContentChildren(self, g: List[int], s: List[int]) -> int:
#         # Sort the greed factors and cookie sizes in ascending order
#         g.sort()
#         s.sort()
#
#         content_children = 0
#         i = 0  # Index for greed factors
#         j = 0  # Index for cookie sizes
#
#         while i < len(g) and j < len(s):
#             if s[j] >= g[i]:
#                 # If the current cookie size is sufficient for the current child
#                 content_children += 1
#                 i += 1  # Move to the next child
#             j += 1  # Move to the next cookie
#
#         return content_children
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findContentChildren(self, g: List[int], s: List[int]) -> int:
#         g.sort()
#         s.sort()
#         l = len(g) - 1
#         m = len(s) - 1
#         count = 0
#         while l>-1 and m>-1:
#             if g[l]<=s[m]:
#                 l-=1
#                 m-=1
#                 count+=1
#             else:
#                 l-=1
#         return count


# 2 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/convert-an-array-into-a-2d-array-with-conditions/?envType=daily-question&envId=2024-01-02 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# from typing import List
#
# class Solution:
#     def findMatrix(self, nums: List[int]) -> List[List[int]]:
#         result = []
#         seen = set()
#         row_dict = defaultdict(list)
#
#         for num in nums:
#             added = False
#             for row in result:
#                 if num not in row:
#                     row.append(num)
#                     seen.add(num)
#                     added = True
#                     break
#
#             if not added:
#                 result.append([num])
#                 seen.add(num)
#
#         return result



# 3 JANUARY 2024
# ~~~~~~~~~~~~~ 2125. Number of Laser Beams in a Bank ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/number-of-laser-beams-in-a-bank/?envType=daily-question&envId=2024-01-03 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ YOUTUBE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def numberOfBeam(self, bank: List[str]) -> int:
#         prev = bank[0].count("1")
#         res = 0
#
#         for i in range(1, len(bank)):
#             curr = bank[i].count("1")
#             res += (prev * curr)
#             if curr:
#                 prev = curr
#         return res



# 4 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/minimum-number-of-operations-to-make-array-empty/description/?envType=daily-question&envId=2024-01-04 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minOperations(self, nums: List[int]) -> int:
#         count = Counter(nums)
#         res = 0
#         for n, c in count.items():
#             if c==1:
#                 return -1
#             res += math.ceil(c/3)
#         return res



# 5 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/longest-increasing-subsequence/description/?envType=daily-question&envId=2024-01-05 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# from typing import List
# class Solution:
#     def lengthOfLIS(self, nums: List[int]) -> int:
#         if not nums:
#             return 0
#         n = len(nums)
#         dp = [1] * n
#         for i in range(1, n):
#             for j in range(i):
#                 if nums[i] > nums[j]:
#                     dp[i] = max(dp[i], dp[j] + 1)
#         return max(dp)



# 6 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/maximum-profit-in-job-scheduling/?envType=daily-question&envId=2024-01-06 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
#         jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])  # Sort jobs by end time
#         n = len(jobs)
#
#         dp = [0] * n  # dp[i] represents the maximum profit achievable until job i
#
#         for i in range(n):
#             # Use binary search to find the latest non-overlapping job
#             prev_job_idx = self.binarySearch(jobs, i)
#
#             # Calculate the maximum profit for the current job
#             include_current = dp[prev_job_idx] + jobs[i][2]
#             exclude_current = dp[i - 1] if i > 0 else 0
#             dp[i] = max(include_current, exclude_current)
#
#         return dp[-1]
#
#     def binarySearch(self, jobs, current_idx):
#         low, high = 0, current_idx - 1
#
#         while low <= high:
#             mid = (low + high) // 2
#             if jobs[mid][1] <= jobs[current_idx][0]:
#                 if jobs[mid + 1][1] <= jobs[current_idx][0]:
#                     low = mid + 1
#                 else:
#                     return mid
#             else:
#                 high = mid - 1
#
#         return -1



# 7 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/?envType=daily-question&envId=2024-01-07 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def numberOfArithmeticSlices(self, nums: List[int]) -> int:
#         n = len(nums)
#         total_count = 0  # Total count of arithmetic subsequences
#
#         # dp[i][diff] represents the count of arithmetic subsequences ending at index i with common difference diff
#         dp = [{} for _ in range(n)]
#
#         for i in range(1, n):
#             for j in range(i):
#                 diff = nums[i] - nums[j]
#
#                 # The count of subsequences ending at index j with common difference diff
#                 prev_count = dp[j].get(diff, 0)
#
#                 # Update the count of subsequences ending at index i with common difference diff
#                 dp[i][diff] = dp[i].get(diff, 0) + prev_count + 1
#
#                 # Update the total count with the count of subsequences ending at index i with common difference diff
#                 total_count += prev_count
#
#         return total_count


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right



# 8 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/range-sum-of-bst/?envType=daily-question&envId=2024-01-08 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
#         def dfs(node):
#             if not node:
#                 return 0
#
#             # Check if the node's value is in the specified range
#             if low <= node.val <= high:
#                 result = node.val
#             else:
#                 result = 0
#
#             # Recursively process the left and right subtrees
#             result += dfs(node.left)
#             result += dfs(node.right)
#
#             return result
#
#         return dfs(root)



# 9 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/leaf-similar-trees/?envType=daily-question&envId=2024-01-09 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
#         def dfs(node, leaves):
#             if not node:
#                 return
#
#             # If the node is a leaf, add its value to the list
#             if not node.left and not node.right:
#                 leaves.append(node.val)
#
#             # Recursively process the left and right subtrees
#             dfs(node.left, leaves)
#             dfs(node.right, leaves)
#
#         # Collect leaf values for both trees
#         leaves1, leaves2 = [], []
#         dfs(root1, leaves1)
#         dfs(root2, leaves2)
#
#         # Compare the leaf value sequences
#         return leaves1 == leaves2



# 10 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/?envType=daily-question&envId=2024-01-10 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#   def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
#     ans = -1
#     graph = self._getGraph(root)
#     q = collections.deque([start])
#     seen = {start}
#
#     while q:
#       ans += 1
#       for _ in range(len(q)):
#         u = q.popleft()
#         if u not in graph:
#           continue
#         for v in graph[u]:
#           if v in seen:
#             continue
#           q.append(v)
#           seen.add(v)
#
#     return ans
#
#   def _getGraph(self, root: Optional[TreeNode]) -> Dict[int, List[int]]:
#     graph = collections.defaultdict(list)
#     q = collections.deque([(root, -1)])  # (node, parent)
#
#     while q:
#       node, parent = q.popleft()
#       if parent != -1:
#         graph[parent].append(node.val)
#         graph[node.val].append(parent)
#       if node.left:
#         q.append((node.left, node.val))
#       if node.right:
#         q.append((node.right, node.val))
#
#     return graph


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right



# 11 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/?envType=daily-question&envId=2024-01-11 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
#         def dfs(node, min_val, max_val):
#             if not node:
#                 return max_val - min_val
#
#             # Update min and max values along the current path
#             min_val = min(min_val, node.val)
#             max_val = max(max_val, node.val)
#
#             # Recursively calculate differences for left and right subtrees
#             left_diff = dfs(node.left, min_val, max_val)
#             right_diff = dfs(node.right, min_val, max_val)
#
#             # Return the maximum difference along the current path
#             return max(left_diff, right_diff)
#
#         # Start DFS with initial values for min and max set to the root's value
#         return dfs(root, root.val, root.val)



# 12 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/determine-if-string-halves-are-alike/description/?envType=daily-question&envId=2024-01-12 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def halvesAreAlike(self, s: str) -> bool:
#         def count_vowels(s):
#             vowels = set("aeiouAEIOU")
#             count = 0
#             for char in s:
#                 if char in vowels:
#                     count += 1
#             return count
#
#         length = len(s)
#         mid = length // 2
#
#         # Split the string into two halves
#         a = s[:mid]
#         b = s[mid:]
#
#         # Count vowels in both halves
#         count_a = count_vowels(a)
#         count_b = count_vowels(b)
#
#         # Check if the counts are equal
#         return count_a == count_b

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def halvesAreAlike(self, s: str) -> bool:
#         vowelset = set("aeiouAEIOU")
#
#         return sum(c in vowelset for c in s[:len(s) // 2]) == sum(c in vowelset for c in s[len(s) // 2:])



# 13 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/description/?envType=daily-question&envId=2024-01-13 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minSteps(self, s: str, t: str) -> int:
#         # Count occurrences of characters in both strings
#         s_count = Counter(s)
#         t_count = Counter(t)
#
#         # Find the difference in counts for each character
#         differences = s_count - t_count
#
#         # Sum of absolute differences gives the minimum number of steps
#         return sum(abs(diff) for diff in differences.values())



# 14 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/determine-if-two-strings-are-close/description/?envType=daily-question&envId=2024-01-14 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

# from collections import Counter
#
#
# class Solution:
#     def closeStrings(self, word1: str, word2: str) -> bool:
#         # Check if the sets of characters are the same
#         if set(word1) != set(word2):
#             return False
#
#         # Check if the frequencies of characters are the same
#         freq1 = Counter(word1)
#         freq2 = Counter(word2)
#
#         # Check if the frequencies of frequencies are the same
#         return sorted(freq1.values()) == sorted(freq2.values())


# 15 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/find-players-with-zero-or-one-losses/?envType=daily-question&envId=2024-01-15 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
#         won_matches = set()
#         lost_matches = {}
#
#         # Update sets and dictionaries based on match outcomes
#         for winner, loser in matches:
#             won_matches.add(winner)
#             lost_matches[loser] = lost_matches.get(loser, 0) + 1
#
#         # Find players who have not lost any matches
#         not_lost_any = list(won_matches - set(lost_matches.keys()))
#
#         # Find players who have lost exactly one match
#         lost_exactly_one = [player for player, lost_count in lost_matches.items() if lost_count == 1]
#
#         # Sort the results in increasing order
#         not_lost_any.sort()
#         lost_exactly_one.sort()
#
#         return [not_lost_any, lost_exactly_one]



# 16 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/insert-delete-getrandom-o1/?envType=daily-question&envId=2024-01-16 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# import random
#
# class RandomizedSet:
#
#     def __init__(self):
#         self.elements = []  # List to store elements
#         self.indices = {}   # Dictionary to store indices of elements
#
#     def insert(self, val: int) -> bool:
#         if val in self.indices:
#             return False  # Element already present, insertion failed
#         self.elements.append(val)
#         self.indices[val] = len(self.elements) - 1
#         return True
#
#     def remove(self, val: int) -> bool:
#         if val not in self.indices:
#             return False  # Element not present, removal failed
#         last_element, idx = self.elements[-1], self.indices[val]
#         self.elements[idx], self.indices[last_element] = last_element, idx
#         self.elements.pop()
#         del self.indices[val]
#         return True
#
#     def getRandom(self) -> int:
#         return random.choice(self.elements)



# 17 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/unique-number-of-occurrences/description/?envType=daily-question&envId=2024-01-17 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ YOUTUBE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# from collections import Counter
# from typing import List
# class Solution:
#     def uniqueOccurrences(self, arr: List[int]) -> bool:
#         # Create a Counter object that counts occurrences of each element in the array.
#         element_count = Counter(arr)
#
#         # Convert the values of the Counter (which represent the occurrences of each unique element) to a set.
#         # This will remove any duplicate counts.
#         unique_occurrences = set(element_count.values())
#
#         # Check if the number of unique occurrences is equal to the number of unique elements.
#         # If they are equal, it means that no two elements have the same number of occurrences.
#         return len(unique_occurrences) == len(element_count)



# 18 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/climbing-stairs/submissions/1149892052/?envType=daily-question&envId=2024-01-18 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def climbStairs(self, n: int) -> int:
#         if n <= 2:
#             return n
#         # Initialize an array to store the number of ways to climb to each step
#         dp = [0] * (n + 1)
#         # Base cases
#         dp[1] = 1
#         dp[2] = 2
#         # Fill the array using the recurrence relation
#         for i in range(3, n + 1):
#             dp[i] = dp[i - 1] + dp[i - 2]
#         return dp[n]