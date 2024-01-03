# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/defanging-an-ip-address/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def defangIPaddr(self, address: str) -> str:
#         return address.replace('.', '[.]')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/build-array-from-permutation/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def buildArray(self, nums: List[int]) -> List[int]:
#         new_list = []
#         for i in nums:
#             new_list.append(nums[i])
#         return new_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/final-value-of-variable-after-performing-operations/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def finalValueAfterOperations(self, operations: List[str]) -> int:
#         count = 0
#         for i in operations :
#             if "++" in operations[i]:
#                 count +=1
#             elif "--" in operations[i]:
#                 count -=1
#         return count


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/maximum-score-after-splitting-a-string/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/path-crossing/description/?envType=daily-question&envId=2023-12-23 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def isPathCrossing(self, path: str) -> bool:
#         directions = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
#         visited = {(0, 0)}
#         x, y = 0, 0
#
#         for direction in path:
#             dx, dy = directions[direction]
#             x, y = x + dx, y + dy
#
#             if (x, y) in visited:
#                 return True
#
#             visited.add((x, y))
#
#         return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/decode-ways/?envType=daily-question&envId=2023-12-25 |
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# def num_decodings(s):
#     if not s or s[0] == '0':
#         return 0
#
#     n = len(s)
#     dp = [0] * (n + 1)
#     dp[0] = 1
#     dp[1] = 1
#
#     for i in range(2, n + 1):
#         # Check if the current digit is not '0'
#         if s[i - 1] != '0':
#             dp[i] += dp[i - 1]
#
#         # Check if the previous two digits form a valid mapping
#         two_digit = int(s[i - 2:i])
#         if 10 <= two_digit <= 26:
#             dp[i] += dp[i - 2]
#
#     return dp[n]
#
# # Example usage:
# encoded_message = "11106"
# ways_to_decode = num_decodings(encoded_message)
# print(ways_to_decode)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def numDecodings(self, s: str) -> int:
#
#         def check_double(s):
#             if s[0] == '0':
#                 return False
#             if s[0] == '1':
#                 return True
#             if s[0] == '2':
#                 if s[1] <='6':
#                     return True
#                 else:
#                     return False
#             return False
#         def check_single(s):
#             if s == '0':
#                 return False
#             return True
#
#         n = len(s)
#         dp = [0 for _ in range(n)]
#         for i in range(n):
#             if check_single(s[i]):
#                 if i-1 >= 0:
#                     dp[i] += dp[i-1]
#                 else:
#                     dp[i] += 1
#             if i-1 >= 0 and check_double(s[i-1:i+1]):
#                 if i-2 >= 0:
#                     dp[i] += dp[i-2]
#                 else:
#                     dp[i] += 1
#         return dp[-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/final-value-of-variable-after-performing-operations/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def finalValueAfterOperations(self, operations: List[str]) -> int:
#         count = 0
#         for i in range(len(operations)):
#             if "--" in operations[i]:
#                 count -=1
#             elif "++" in operations[i]:
#                 count +=1
#         return count


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/two-sum/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# #~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         for i in range(len(nums)):
#             num1 = nums[i]
#             num2 = nums[i+1]
#             if num1 + num2 == target:
#                 return [num1, num2]

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         n = len(nums)
#         for i in range(n - 1):
#             for j in range(i + 1, n):
#                 num1 = nums[i]
#                 num2 = nums[j]
#                 if num1 + num2 == target:
#                     return [i, j]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/longest-substring-without-repeating-characters/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

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


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/palindrome-number/description/ |
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # ~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         x_str = str(x)
#         return x_str == x_str[::-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/convert-the-temperature/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
"""
Kelvin = Celsius + 273.15
Fahrenheit = Celsius * 1.80 + 32.00
"""
# class Solution:
#     def convertTemperature(self, celsius: float) -> List[float]:
#         kelvin = celsius + 273.15
#         fahrenheit = celsius * 1.80 + 32.00
#         return kelvin, fahrenheit


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minPartitions(self, n: str) -> int:
#         # Находим максимальную цифру в числе n
#         max_digit = max(map(int, n))
#         return max_digit


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/concatenation-of-array/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def getConcatenation(self, nums: List[int]) -> List[int]:
#         return nums + nums.copy()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/reverse-integer/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def reverse(self, x: int) -> int:
#         if x <= -1:
#             x = x * -1
#             x = str(x)[::-1]
#             x = int(x)*-1
#             return x
#         elif x > 0:
#             x = str(x)[::-1]
#             x = int(x)
#             return x
# ~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def reverse(self, x: int) -> int:
#         INT_MAX = 2 ** 31 - 1
#         INT_MIN = -2 ** 31
#
#         # Convert the integer to a string
#         str_x = str(x)
#
#         # Handle the sign and reverse the string
#         if x < 0:
#             reversed_str = '-' + str_x[:0:-1]
#         else:
#             reversed_str = str_x[::-1]
#
#         # Convert the reversed string back to an integer
#         reversed_int = int(reversed_str)
#
#         # Check if the result is within the 32-bit integer range
#         if reversed_int > INT_MAX or reversed_int < INT_MIN:
#             return 0
#
#         return reversed_int


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # # https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/?envType=daily-question&envId=2023-l2-27 |
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minCost(self, colors: str, neededTime: List[int]) -> int:
#         n = len(colors)
#         new_colors = []
#         colors = list(colors)
#         for i in range(n - l):
#             for j in range(i + l, n):
#                 colori = colors[i]
#                 colorj = colors[j]
#                 if colori != colorj:
#                     new_colors.append(colori)
#                     new_colors.append(colorj)
#                     new_colors_sum = len(new_colors)
#                     return new_colors_sum
#                 else:
#                     colors.remove(colori)
#
#     minCost("abaac", [l, 2, 3, 4, 5])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ YOU TUBE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def minCost(self, colors: str, neededTime: List[int]) -> int:
#         l = res = 0
#         for r in range(1, len(colors)):
#             if colors[l] == colors[r]:
#                 if neededTime[l] < neededTime[r]:
#                     res += neededTime[l]
#                     l = r
#                 else:
#                     res += neededTime[r]
#             else:
#                 l = r
#         return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/median-of-two-sorted-arrays/submissions/1129949277/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
#         if len(nums1) > len(nums2):
#             nums1, nums2 = nums2, nums1
#
#         x, y = len(nums1), len(nums2)
#         low, high = 0, x
#
#         while low <= high:
#             partitionX = (low + high) // 2
#             partitionY = (x + y + 1) // 2 - partitionX
#
#             maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
#             minX = float('inf') if partitionX == x else nums1[partitionX]
#
#             maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
#             minY = float('inf') if partitionY == y else nums2[partitionY]
#
#             if maxX <= minY and maxY <= minX:
#                 if (x + y) % 2 == 0:
#                     return (max(maxX, maxY) + min(minX, minY)) / 2.0
#                 else:
#                     return max(maxX, maxY)
#             elif maxX > minY:
#                 high = partitionX - 1
#             else:
#                 low = partitionX + 1
#
# Example usage:
# solution = Solution()
#
# nums1_1, nums2_1 = [1, 3], [2]
# nums1_2, nums2_2 = [1, 2], [3, 4]
#
# result1 = solution.findMedianSortedArrays(nums1_1, nums2_1)
# result2 = solution.findMedianSortedArrays(nums1_2, nums2_2)
#
# print("Example 1:", result1)
# print("Example 2:", result2)


# Optimal solution
from typing import List

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def solve():
#     with open('user.out', 'w') as f:
#         data = map(loads, stdin)
#         while True:
#             try:
#                 nums1, nums2 = next(data), next(data)
#             except StopIteration:
#                 break
#
#             lentotal = (len1 := len(nums1)) + (len2 := len(nums2))
#             if len1 < len2:
#                 nums1, nums2, len1, len2 = nums2, nums1, len2, len1
#             halflentotal, halflen1, halflen2, odd = lentotal // 2, len1 / 2, len2 / 2, lentotal % 2 == 1
#             try:
#                 for a, b, lena1, lenb, start, end in (
#                         (nums2, nums1, len2 - 1, len1, 0, len2 - 1),
#                         (nums1, nums2, len1 - 1, len2, int(halflen1 - halflen2), int(halflen1 + halflen2)),
#                 ):
#                     while start <= end:
#                         idxa = start + (end - start) // 2
#                         aval, idxb = \
#                             a[idxa], \
#                                 0 if (diff := (diff if (diff := halflentotal - idxa) < lenb else lenb)) < 0 else diff
#                         if aval > (inf if idxa == lena1 else a[idxa + 1]) or \
#                                 aval > (inf if idxb == lenb else b[idxb]):
#                             end = idxa - 1
#                         elif aval < (avalprev := -inf if idxa == 0 else a[idxa - 1]) or \
#                                 aval < (bvalprev := -inf if idxb == 0 else b[idxb - 1]):
#                             start = idxa + 1
#                         else:
#                             result = aval if odd else (aval + (avalprev if avalprev > bvalprev else bvalprev)) / 2
#                             raise StopIteration
#             except StopIteration:
#                 pass
#
#             print(f"{result:.5f}", file=f)
#
#
# solve()
# exit()

# class Solution:
#     def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
#
#         # We find the total length of the list
#         n, m = len(nums1), len(nums2)
#         total_len = n + m
#
#         # We calculate the index of the middle of the list
#         cur, mid = 0, total_len / 2
#         is_odd = 0
#
#         # We mark if it is odd, we are going to have to divide by the two middle elements if it is even
#         if not mid.is_integer():
#             is_odd += 1
#
#         # we do the actual calculation for us to be able to iterate to the middle of the list
#         mid = (total_len // 2) + 1
#
#         i, j, cur_list = 0, 0, []
#
#         # We create the first half of the list + 1 to be able to do the division for the median
#         while cur < mid:
#             if i >= n:
#                 cur += 1
#                 cur_list.append(nums2[j])
#                 j += 1
#                 continue
#             if j >= m:
#                 cur += 1
#                 cur_list.append(nums1[i])
#                 i += 1
#                 continue
#             if nums2[j] > nums1[i]:
#                 cur += 1
#                 cur_list.append(nums1[i])
#                 i += 1
#             else:
#                 cur += 1
#                 cur_list.append(nums2[j])
#                 j += 1
#
#         # We return the appropriate values at the end of the list
#         if not is_odd:
#             return (cur_list[-1] + cur_list[-2]) / 2
#
#         else:
#             return cur_list[-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/description/?envType=daily-question&envId=2023-12-26|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def numRollsToTarget(self, n: int, k: int, target: int) -> int:
#         MOD = 10 ** 9 + 7
#
#         # dp[i][j]: number of ways to get sum j with i dice
#         dp = [[0] * (target + 1) for _ in range(n + 1)]
#
#         # Base case: With 0 dice, the sum 0 is the only way
#         dp[0][0] = 1
#
#         for i in range(1, n + 1):
#             for j in range(1, target + 1):
#                 # Try all possible outcomes for the current die
#                 for face in range(1, k + 1):
#                     if j - face >= 0:
#                         dp[i][j] = (dp[i][j] + dp[i - 1][j - face]) % MOD
#
#         return dp[n][target]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/string-compression-ii/submissions/1130341212/?envType=daily-question&envId=2023-12-28 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ YOU TUBE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
#         cache = {}
#         def count(i, k, prev, prev_cnt):
#             if (i, k, prev, prev_cnt) in cache:
#                 return cache[(i, k, prev, prev_cnt)]
#             if k < 0:
#                 return float("inf")
#             if i == len(s):
#                 return 0
#
#             if s[i] == prev:
#                 incr = 1 if prev_cnt in [1, 9, 99] else 0
#                 res = incr + count(i + 1, k, prev, prev_cnt + 1)
#             else:
#                 res = min(
#                     count(i + 1, k - 1, prev, prev_cnt), # delete s[i]
#                     1 + count(i + 1, k, s[i], 1) # dont delete
#                 )
#             cache[(i, k, prev, prev_cnt)] = res
#             return res
#         return count(0, k, "", 0)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/zigzag-conversion/submissions/1130703655/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def convert(self, s: str, numRows: int) -> str:
#         if numRows == 1 or len(s) <= numRows:
#             return s
#
#         rows = [''] * numRows
#         current_row, going_down = 0, False
#
#         for char in s:
#             rows[current_row] += char
#
#             if current_row == 0 or current_row == numRows - 1:
#                 going_down = not going_down
#
#             current_row += 1 if going_down else -1
#
#         result = ''.join(rows)
#         return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/?envType=daily-question&envId=2023-12-29 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
#         n = len(jobDifficulty)
#         if n < d:
#             return -1
#
#         # dp[i][j]: minimum difficulty to schedule i jobs in j days
#         dp = [[float('inf')] * (d + 1) for _ in range(n + 1)]
#         dp[0][0] = 0
#
#         for i in range(1, n + 1):
#             for k in range(1, d + 1):
#                 max_difficulty = 0
#                 for j in range(i - 1, -1, -1):
#                     max_difficulty = max(max_difficulty, jobDifficulty[j])
#                     dp[i][k] = min(dp[i][k], dp[j][k - 1] + max_difficulty)
#
#         return dp[n][d] if dp[n][d] != float('inf') else -1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/contest/weekly-contest-378/problems/check-if-bitwise-or-has-trailing-zeros/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ WEEKLY CONTEST ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def hasTrailingZeros(self, nums: List[int]) -> bool:
#         # Iterate through all pairs of elements in the array
#         for i in range(len(nums)):
#             for j in range(i + 1, len(nums)):
#                 # Check if the bitwise OR of nums[i] and nums[j] has at least one trailing zero
#                 if (nums[i] | nums[j]) & 1 == 0:
#                     return True
#         return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/largest-substring-between-two-equal-characters/?envType=daily-question&envId=2023-12-31 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |
# # https://leetcode.com/problems/longest-palindromic-substring/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |
# class Solution:
# def longestPalindrome(self, s: str) -> str:
# def is_polindrom_func(r_str):
#     return r_str == r_str[::-1]
#
#
# s = "babad"
# is_polindrom = is_polindrom_func(s)
# polindrom = ""
# if is_polindrom == True:
#     print(s)
# elif is_polindrom == False:
#     for i in range(len(s) - 1):
#         polindrom = i
#         ss = is_polindrom_func(polindrom)
#         if ss == True:
#             print(polindrom)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def longestPalindrome(self, s: str) -> str:
#         n = len(s)
#         start = 0
#         max_length = 1
#         is_palindrome = [[False] * n for _ in range(n)]
#
#         # All substrings of length 1 are palindromes
#         for i in range(n):
#             is_palindrome[i][i] = True
#
#         # Check substrings of length 2
#         for i in range(n - 1):
#             if s[i] == s[i + 1]:
#                 is_palindrome[i][i + 1] = True
#                 start = i
#                 max_length = 2
#
#         # Check substrings of length 3 or more
#         for length in range(3, n + 1):
#             for i in range(n - length + 1):
#                 j = i + length - 1
#                 if is_palindrome[i + 1][j - 1] and s[i] == s[j]:
#                     is_palindrome[i][j] = True
#                     start = i
#                     max_length = length
#
#         return s[start:start + max_length]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class Solution:
#     def longestPalindrome(self, s: str) -> str:
#         res=""
#         for i in range(len(s)):
#             r=i
#             l=i
#             while l>=0 and r<=len(s)-1  and s[r]==s[l]:
#                 if len(res)<len(s[l:r+1]):
#                     res=s[l:r+1]
#                 r+=1
#                 l-=1
#             r=i+1
#             l=i
#             while l>=0 and r<=len(s)-1  and s[r]==s[l]:
#                 if len(res)<len(s[l:r+1]):
#                     res=s[l:r+1]
#                 r+=1
#                 l-=1
#         return res


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!~~~~~|
# # https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/?envType=daily-question&envId=2023-12-29 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
#         n = len(jobDifficulty)
#         if n < d:
#             return -1
#
#         # dp[i][j]: minimum difficulty to schedule i jobs in j days
#         dp = [[float('inf')] * (d + 1) for _ in range(n + 1)]
#         dp[0][0] = 0
#
#         for i in range(1, n + 1):
#             for k in range(1, d + 1):
#                 max_difficulty = 0
#                 for j in range(i - 1, -1, -1):
#                     max_difficulty = max(max_difficulty, jobDifficulty[j])
#                     dp[i][k] = min(dp[i][k], dp[j][k - 1] + max_difficulty)
#
#         return dp[n][d] if dp[n][d] != float('inf') else -1


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



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/?envType=daily-question&envId=2023-12-30 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def makeEqual(self, words: List[str]) -> bool:
#         char_count = Counter("".join(words))
#         return all(count % len(words) == 0 for count in char_count.values())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def makeEqual(self, words: List[str]) -> bool:
#         s = "".join(words)
#         for letter in set(s):
#             if s.count(letter) % len(words) != 0:
#                 return False
#         return True



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



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/submissions/1133785283/?envType=daily-question&envId=2023-12-30 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def makeEqual(self, words: List[str]) -> bool:
#         char_count = Counter("".join(words))
#         return all(count % len(words) == 0 for count in char_count.values())



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



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/roman-to-integer/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def romanToInt(self, s: str) -> int:
#         rome_numbers = {
#             "I": 1,
#             "V": 5,
#             "X": 10,
#             "L": 50,
#             "C": 100,
#             "D": 500,
#             "M": 1000, }
#         roman_int = 0
#         for i in range(len(s)):
#             if s[i] == "I":
#                 roman_int += 1
#             elif s[i] == "V":
#                 roman_int += 5
#             elif s[i] == "X":
#                 roman_int += 10
#             elif s[i] == "L":
#                 roman_int += 50
#             elif s[i] == "C":
#                 roman_int += 100
#             elif s[i] == "D":
#                 roman_int += 500
#             elif s[i] == "M":
#                 roman_int += 1000
#         return roman_int

# ~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def romanToInt(self, s: str) -> int:
#         roman_values = {
#             'I': 1,
#             'V': 5,
#             'X': 10,
#             'L': 50,
#             'C': 100,
#             'D': 500,
#             'M': 1000
#         }
#         result = 0
#         prev_value = 0
#         for char in s:
#             value = roman_values[char]
#             if value > prev_value:
#                 # If a smaller value precedes a larger value, subtract the smaller value
#                 result += value - 2 * prev_value
#             else:
#                 result += value
#             prev_value = value
#         return result



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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