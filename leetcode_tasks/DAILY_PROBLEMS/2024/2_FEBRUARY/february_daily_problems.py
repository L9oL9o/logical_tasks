# 1 FEBRUARY 2024
# https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description/?envType=daily-question&envId=2024-02-01 |

# class Solution:
#     def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
#         nums.sort()
#         result = []
#         is_possible = True
#         for i in range(0, len(nums) - 2, 3):
#             first, second, third = nums[i], nums[i + 1], nums[i + 2]
#             if third - first <= k:
#                 result.append([first, second, third])
#             else:
#                 is_possible = False
#                 break
#         if not is_possible:
#             return []
#         return result

# 2 FEBRUARY 2024
# https://leetcode.com/problems/sequential-digits/description/?envType=daily-question&envId=2024-02-02

# class Solution:
#     def sequentialDigits(self, low: int, high: int) -> List[int]:
#         t = '123456789'
#         l = []
#         for i in range(len(t)):
#             for j in range(i + 1, len(t) + 1):
#                 if low <= int(t[i:j]) <= high:
#                     l.append(int(t[i:j]))
#         return sorted(l)


# 3 FEBRUARY 2024
# https://leetcode.com/problems/partition-array-for-maximum-sum/description/?envType=daily-question&envId=2024-02-03

# class Solution:
#     def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
#         n = len(arr)
#         dp = [0]*(n+1)
#         for i in range(n):
#             curMax = curSum = 0
#             for j in range(i, max(-1, i-k), -1):
#                 if curMax < arr[j]:
#                     curMax = arr[j]
#                 cur = curMax*(i-j+1)+dp[j]
#                 if curSum < cur:
#                     curSum = cur
#             dp[i+1] = curSum
#         return dp[-1]
#


# 4 FEBRUARY 2024
# https://leetcode.com/problems/minimum-window-substring/description/?envType=daily-question&envId=2024-02-04

# class Solution:
#     def minWindow(self, s: str, t: str) -> str:
#         if len(s) < len(t):
#             return ""
#         needstr = collections.defaultdict(int)
#         for ch in t:
#             needstr[ch] += 1
#         needcnt = len(t)
#         res = (0, float('inf'))
#         start = 0
#         for end, ch in enumerate(s):
#             if needstr[ch] > 0:
#                 needcnt -= 1
#             needstr[ch] -= 1
#             if needcnt == 0:
#                 while True:
#                     tmp = s[start]
#                     if needstr[tmp] == 0:
#                         break
#                     needstr[tmp] += 1
#                     start += 1
#                 if end - start < res[1] - res[0]:
#                     res = (start, end)
#                 needstr[s[start]] += 1
#                 needcnt += 1
#                 start += 1
#         return '' if res[1] > len(s) else s[res[0]:res[1]+1]


# 5 FEBRUARY 2024
# https://leetcode.com/problems/first-unique-character-in-a-string/description/?envType=daily-question&envId=2024-02-05

# class Solution:
#     def firstUniqChar(self, s: str) -> int:
#         char_count = {}
#
#         # Count the frequency of each character in the string
#         for char in s:
#             char_count[char] = char_count.get(char, 0) + 1
#
#         # Iterate through the string to find the first unique character
#         for i in range(len(s)):
#             if char_count[s[i]] == 1:
#                 return i
#
#         # If no unique character is found, return -1
#         return -1


# 6 FEBRUARY 2024
# https://leetcode.com/problems/group-anagrams/description/?envType=daily-question&envId=2024-02-06

# f = open('user.out', 'w')
#
# for strs in map(loads, stdin):
#     w = defaultdict(list)
#     for s in strs:
#         w[''.join(sorted(s))].append(s)
#
#     print(str(w.values())[12:-1].replace("'", '"').replace(" ", ""), file=f)
# exit()


# 7 FEBRUARY 2024
# https://leetcode.com/problems/sort-characters-by-frequency/description/?envType=daily-question&envId=2024-02-07

# class Solution:
#     def frequencySort(self, s: str) -> str:
#         k = Counter(s)
#         k4 = ''
#         # Sort characters by frequency and then by dictionary order
#         sorted_chars = sorted(k, key=lambda x: (-k[x], x))
#         for char in sorted_chars:
#             k4 += char * k[char]
#         return k4


# 8 FEBRUARY 2024
# https://leetcode.com/problems/perfect-squares/description/?envType=daily-question&envId=2024-02-08

# class Solution:
#   def numSquares(self, n: int) -> int:
#     dp = [n] * (n + 1)  # 1^2 x n
#     dp[0] = 0  # no way
#     dp[1] = 1  # 1^2
#     for i in range(2, n + 1):
#       j = 1
#       while j * j <= i:
#         dp[i] = min(dp[i], dp[i - j * j] + 1)
#         j += 1
#     return dp[n]


# 9 FEBRUARY 2024
# https://leetcode.com/problems/largest-divisible-subset/description/?envType=daily-question&envId=2024-02-09

# class Solution:
#   def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
#     n = len(nums)
#     ans = []
#     count = [1] * n
#     prevIndex = [-1] * n
#     maxCount = 0
#     index = -1
#     nums.sort()
#     for i, num in enumerate(nums):
#       for j in reversed(range(i)):
#         if num % nums[j] == 0 and count[i] < count[j] + 1:
#           count[i] = count[j] + 1
#           prevIndex[i] = j
#       if count[i] > maxCount:
#         maxCount = count[i]
#         index = i
#     while index != -1:
#       ans.append(nums[index])
#       index = prevIndex[index]
#     return ans


# 10 FEBRUARY 2024
# https://leetcode.com/problems/palindromic-substrings/description/?envType=daily-question&envId=2024-02-10

# class Solution:
#     def countSubstrings(self, S: str) -> int:
#         ans, n, i = 0, len(S), 0
#         while (i < n):
#             j, k = i - 1, i
#             while k < n - 1 and S[k] == S[k+1]: k += 1
#             ans += (k - j) * (k - j + 1) // 2
#             i, k = k + 1, k + 1
#             while ~j and k < n and S[k] == S[j]:
#                 j, k, ans = j - 1, k + 1, ans + 1
#         return ans


# 11 FEBRUARY 2024
# https://leetcode.com/problems/cherry-pickup-ii/description/?envType=daily-question&envId=2024-02-11

# class Solution:
#     def cherryPickup(self, grid: List[List[int]]) -> int:
#         m, n = len(grid), len(grid[0])
#         if n == 2:
#             return sum(sum(g) for g in grid)
#         elif n == 3:
#             return sum(sum(g) - min(g) for g in grid)
#
#         prev, dirs = [[0] * n for _ in range(n)], {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0),
#                                                    (1, 1)}
#         for i in range(m - 1, -1, -1):
#             cur = [[0] * n for _ in range(n)]
#             for j in range(min(i + 1, n)):
#                 for k in range(max(j, n - i - 1), n):
#                     for dir in dirs:
#                         s, t = j + dir[0], k + dir[1]
#                         if 0 <= s <= t < n and cur[j][k] < prev[s][t]:
#                             cur[j][k] = prev[s][t]
#                     cur[j][k] += grid[i][j] + (0 if j == k else grid[i][k])
#             prev = cur
#         return prev[0][-1]


# 12 FEBRUARY 2024
# https://leetcode.com/problems/majority-element/description/?envType=daily-question&envId=2024-02-12

# import statistics
# f = open("user.out", 'w')
# for line in stdin:
#     l = sorted(map(int, line.rstrip()[1:-1].split(',')))
#     print(l[len(l) // 2], file=f)
# exit(0)


# 13 FEBRUARY 2024
# https://leetcode.com/problems/find-first-palindromic-string-in-the-array/description/?envType=daily-question&envId=2024-02-13

# f = open('user.out', 'w')
# for test in map(loads, stdin):
#     flag = True
#     for word in test:
#         if word == word[::-1]:
#             flag = False
#             print('"' + word + '"', file=f)
#             break
#     if flag: print('""', file=f)
# exit(0)


# 14 FEBRUARY 2024

# 15 FEBRUARY 2024

# 16 FEBRUARY 2024

# 17 FEBRUARY 2024

# 18 FEBRUARY 2024

# 19 FEBRUARY 2024

# 20 FEBRUARY 2024

# 21 FEBRUARY 2024

# 22 FEBRUARY 2024

# 23 FEBRUARY 2024

# 24 FEBRUARY 2024

# 25 FEBRUARY 2024

# 26 FEBRUARY 2024

# 27 FEBRUARY 2024

# 28 FEBRUARY 2024

# 29 FEBRUARY 2024



