# 22 DECEMBER 2023
# https://leetcode.com/problems/maximum-score-after-splitting-a-string/?envType=daily-question&envId=2023-12-22

# class Solution:
#     def maxScore(self, s: str) -> int:
#         max_score = 0
#         zeros_on_left = 0
#         ones_on_right = s.count('1')
#
#         for i in range(len(s) - 1):
#             if s[i] == '0':
#                 zeros_on_left += 1
#             else:
#                 ones_on_right -= 1
#
#             score = zeros_on_left + ones_on_right
#             max_score = max(max_score, score)
#
#         return max_score


# 23 DECEMBER 2023
# https://leetcode.com/problems/path-crossing/submissions/1162739893/?envType=daily-question&envId=2023-12-23
# class Solution:
#     def isPathCrossing(self, path: str) -> bool:
#         arr=[]
#         x = 0
#         y = 0
#         for d in path:
#             arr.append((x,y))
#             if d == "N":
#                 y+=1
#             elif d == "S":
#                 y-=1
#             elif d == "E":
#                 x+=1
#             elif d== "W":
#                 x-=1
#             if (x,y) in arr:
#                 return True
#         return False


# 24 DECEMBER 2023
# MISSED DAY


# 25 DECEMBER 2023
# https://leetcode.com/problems/decode-ways/description/?envType=daily-question&envId=2023-12-25

# class Solution:
#     def numDecodings(self, s: str) -> int:
#         if not s or s[0] == '0':
#             return 0
#         n = len(s)
#         dp = [0] * (n + 1)
#         dp[0] = 1
#         dp[1] = 1
#         for i in range(2, n + 1):
#             # Check if the current digit is not '0'
#             if s[i - 1] != '0':
#                 dp[i] += dp[i - 1]
#             # Check if the previous two digits form a valid mapping
#             two_digit = int(s[i - 2:i])
#             if 10 <= two_digit <= 26:
#                 dp[i] += dp[i - 2]
#         return dp[n]


# 26 DECEMBER 2023
# https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/description/?envType=daily-question&envId=2023-12-26

# class Solution:
#     def numRollsToTarget(self, n: int, k: int, target: int) -> int:
#         MOD = 10 ** 9 + 7
#         # dp[i][j]: number of ways to get sum j with i dice
#         dp = [[0] * (target + 1) for _ in range(n + 1)]
#         # Base case: With 0 dice, the sum 0 is the only way
#         dp[0][0] = 1
#         for i in range(1, n + 1):
#             for j in range(1, target + 1):
#                 # Try all possible outcomes for the current die
#                 for face in range(1, k + 1):
#                     if j - face >= 0:
#                         dp[i][j] = (dp[i][j] + dp[i - 1][j - face]) % MOD
#         return dp[n][target]


# 27 DECEMBER 2023
# https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/?envType=daily-question&envId=2023-12-27

# def minCost0(colors, neededTime):
#     C = colors
#     T = neededTime
#     cj = C[0]
#     maxt = T[0]
#     s = 0
#     for ci, ti in zip(C[1:], T[1:]):
#         #print("1 ci,cj,ti,tj",(ci,cj,ti,maxt,s))
#         if ci != cj:
#             maxt = ti
#             cj = ci
#         else:
#             if ti > maxt:
#                 s += maxt
#                 maxt = ti
#             else:
#                 s += ti
#         #print("2 ci,cj,ti,tj",(ci,cj,ti,maxt,s))
#     return s
# class Solution:
#     def minCost(self, colors: str, neededTime: List[int]) -> int:
#         return minCost0(colors, neededTime)


# 28 DECEMBER 2023
# https://leetcode.com/problems/string-compression-ii/description/?envType=daily-question&envId=2023-12-28

# class Solution:
#     def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
#         e = lambda x: 1 if x < 2 else 2 if x < 10 else 3 if x < 100 else 4
#         @cache
#         def f(i, k):
#             if i < 0: return 0
#             r = f(i-1, k-1) if k else 9e9
#             x = 0
#             for j in range(i, -1, -1):
#                 if s[i] == s[j]: x += 1
#                 elif k == 0: return r
#                 else: k -= 1
#                 r = min(r, f(j-1, k) + e(x))
#             return r
#         return f(len(s)-1, k)


# 29 DECEMBER 2023
# https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/?envType=daily-question&envId=2023-12-29

# class Solution:
#     def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
#         n = len(jobDifficulty)
#         if n < d:
#             return -1
#         # dp[i][j]: minimum difficulty to schedule i jobs in j days
#         dp = [[float('inf')] * (d + 1) for _ in range(n + 1)]
#         dp[0][0] = 0
#         for i in range(1, n + 1):
#             for k in range(1, d + 1):
#                 max_difficulty = 0
#                 for j in range(i - 1, -1, -1):
#                     max_difficulty = max(max_difficulty, jobDifficulty[j])
#                     dp[i][k] = min(dp[i][k], dp[j][k - 1] + max_difficulty)
#         return dp[n][d] if dp[n][d] != float('inf') else -1


# 30 DECEMBER 2023
# https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/description/?envType=daily-question&envId=2023-12-30

# class Solution:
#     def makeEqual(self, words: List[str]) -> bool:
#         char_count = Counter("".join(words))
#         return all(count % len(words) == 0 for count in char_count.values())


# 31 DECEMBER 2023
# https://leetcode.com/problems/largest-substring-between-two-equal-characters/?envType=daily-question&envId=2023-12-31

# class Solution:
#     def maxLengthBetweenEqualCharacters(self, s: str) -> int:
#         first_occurrence = {}
#         max_length = -1
#
#         for i, char in enumerate(s):
#             if char in first_occurrence:
#                 max_length = max(max_length, i - first_occurrence[char] - 1)
#             else:
#                 first_occurrence[char] = i
#
#         return max_length
