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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/median-of-two-sorted-arrays/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/longest-common-prefix/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def longestCommonPrefix(self, strs: List[str]) -> str:
#         if not strs:
#             return ""
#         # Sort the strings to compare only the first and last strings
#         strs.sort()
#         first_str = strs[0]
#         last_str = strs[-1]
#         common_prefix = []
#         for i in range(len(first_str)):
#             if i < len(last_str) and first_str[i] == last_str[i]:
#                 common_prefix.append(first_str[i])
#             else:
#                 break
#         return ''.join(common_prefix)


# ~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/regular-expression-matching/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         # Create a 2D DP array to store matching results
#         dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
#
#         # Empty string and empty pattern match
#         dp[0][0] = True
#
#         # Handle patterns with '*'
#         for j in range(1, len(p) + 1):
#             if p[j - 1] == '*':
#                 dp[0][j] = dp[0][j - 2]
#
#         # Fill in the DP array
#         for i in range(1, len(s) + 1):
#             for j in range(1, len(p) + 1):
#                 if p[j - 1] == s[i - 1] or p[j - 1] == '.':
#                     dp[i][j] = dp[i - 1][j - 1]
#                 elif p[j - 1] == '*':
#                     dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] if s[i - 1] == p[j - 2] or p[j - 2] == '.' else False)
#
#         return dp[-1][-1]


# ~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/merge-k-sorted-lists |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# from queue import PriorityQueue
# from typing import List, Optional
#
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
# class Solution:
#     def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
#         # Create a priority queue (min-heap) to keep track of the current minimum node
#         min_heap = PriorityQueue()
#
#         # Add the first node from each list to the min-heap
#         for i, lst in enumerate(lists):
#             if lst:
#                 min_heap.put((lst.val, i, lst))
#
#         # Dummy node to simplify the code
#         dummy = ListNode()
#         current = dummy
#
#         while not min_heap.empty():
#             val, index, node = min_heap.get()
#             current.next = node
#             current = current.next
#
#             # Move to the next node in the list
#             if node.next:
#                 min_heap.put((node.next.val, index, node.next))
#
#         return dummy.next


# ~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/substring-with-concatenation-of-all-words/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findSubstring(self, s: str, words: List[str]) -> List[int]:
#         if not s or not words:
#             return []
#
#         word_len = len(words[0])
#         word_count = len(words)
#         total_len = word_len * word_count
#         word_freq = Counter(words)
#
#         result = []
#
#         for i in range(word_len):
#             left, right = i, i
#             current_window = Counter()
#
#             while right + word_len <= len(s):
#                 current_word = s[right:right + word_len]
#                 right += word_len
#                 current_window[current_word] += 1
#
#                 while current_window[current_word] > word_freq[current_word]:
#                     current_window[s[left:left + word_len]] -= 1
#                     left += word_len
#
#                 if right - left == total_len:
#                     result.append(left)
#
#         return result


# ~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/longest-valid-parentheses/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def longestValidParentheses(self, s: str) -> int:
#         if not s:
#             return 0
#
#         n = len(s)
#         dp = [0] * n
#         max_len = 0
#
#         for i in range(1, n):
#             if s[i] == ')':
#                 if s[i - 1] == '(':
#                     dp[i] = dp[i - 2] + 2 if i >= 2 else 2
#                 elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == '(':
#                     dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2 if i - dp[i - 1] >= 2 else dp[i - 1] + 2
#
#                 max_len = max(max_len, dp[i])
#
#         return max_len
#
# # Example usage:
# solution = Solution()
#
# # Example 1
# s1 = "(()"
# result1 = solution.longestValidParentheses(s1)
# # Output: 2
#
# # Example 2
# s2 = ")()())"
# result2 = solution.longestValidParentheses(s2)
# # Output: 4
#
# # Example 3
# s3 = ""
# result3 = solution.longestValidParentheses(s3)
# # Output: 0


# ~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/trapping-rain-water/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def trap(self, height: List[int]) -> int:
#         if not height or len(height) < 3:
#             return 0
#
#         n = len(height)
#         left, right = 0, n - 1
#         left_max, right_max = height[left], height[right]
#         water = 0
#
#         while left < right:
#             left_max = max(left_max, height[left])
#             right_max = max(right_max, height[right])
#
#             if left_max < right_max:
#                 water += left_max - height[left]
#                 left += 1
#             else:
#                 water += right_max - height[right]
#                 right -= 1
#
#         return water


# ~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~|
# # https://leetcode.com/problems/n-queens/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~|
# class Solution:
#     def solveNQueens(self, n: int) -> List[List[str]]:
#         def is_valid(board, row, col, n):
#             # Check if there is a queen in the same column
#             for i in range(row):
#                 if board[i][col] == 'Q':
#                     return False
#
#             # Check if there is a queen in the left diagonal
#             for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
#                 if board[i][j] == 'Q':
#                     return False
#
#             # Check if there is a queen in the right diagonal
#             for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
#                 if board[i][j] == 'Q':
#                     return False
#
#             return True
#
#         def solve(board, row, n, result):
#             if row == n:
#                 result.append(["".join(row) for row in board])
#                 return
#
#             for col in range(n):
#                 if is_valid(board, row, col, n):
#                     board[row][col] = 'Q'
#                     solve(board, row + 1, n, result)
#                     board[row][col] = '.'
#
#         result = []
#         board = [['.'] * n for _ in range(n)]
#         solve(board, 0, n, result)
#         return result


# ~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/?envType=daily-question&envId=2024-01-07 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/contest/weekly-contest-380/problems/find-beautiful-indices-in-the-given-array-i/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
#         result = []
#
#         for i in range(len(s) - len(a) + 1):
#             if s[i:i + len(a)] == a:
#                 for j in range(max(0, i - k), min(len(s) - len(b) + 1, i + k + 1)):
#                     if s[j:j + len(b)] == b and abs(i - j) <= k:
#                         result.append(i)
#                         break
#
#         return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/determine-if-two-strings-are-close/description/?envType=daily-question&envId=2024-01-14 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/minimum-falling-path-sum/description/?envType=daily-question&envId=2024-01-19 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ YOUTUBE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def minFallingPathSum(self, matrix: List[List[int]]) -> int:
#         N = len(matrix)
#
#         for r in range(1, N):
#             for c in range(N):
#                 mid = matrix[r - 1][c]
#                 left = matrix[r - 1][c - 1] if c > 0 else float("inf")
#                 right = matrix[r - 1][c + 1] if c < N - 1 else float("inf")
#                 matrix[r][c] = matrix[r][c] + min(mid, left, right)
#
#         return min(matrix[-1])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/sum-of-subarray-minimums/?envType=daily-question&envId=2024-01-20 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def sumSubarrayMins(self, arr: List[int]) -> int:
#         mod = 10**9 + 7
#         stack = []
#         result = 0
#         for i, num in enumerate(arr):
#             while stack and arr[stack[-1]] > num:
#                 top = stack.pop()
#                 left = stack[-1] if stack else -1
#                 result += arr[top] * (i - top) * (top - left)
#                 result %= mod
#             stack.append(i)
#         # Process the remaining elements in the stack
#         while stack:
#             top = stack.pop()
#             left = stack[-1] if stack else -1
#             result += arr[top] * (len(arr) - top) * (top - left)
#             result %= mod
#         return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/house-robber/description/?envType=daily-question&envId=2024-01-21 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#   def rob(self, nums: List[int]) -> int:
#     if not nums:
#       return 0
#     if len(nums) == 1:
#       return nums[0]
#
#     # dp[i]:= max money of robbing nums[0..i]
#     dp = [0] * len(nums)
#     dp[0] = nums[0]
#     dp[1] = max(nums[0], nums[1])
#
#     for i in range(2, len(nums)):
#       dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
#
#     return dp[-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/set-mismatch/description/?envType=daily-question&envId=2024-01-22 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findErrorNums(self, nums: List[int]) -> List[int]:
#         count = [0] * len(nums)
#         for i in nums:
#             count[i - 1] += 1
#         return [count.index(2) + 1, count.index(0) + 1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/description/?envType=daily-question&envId=2024-01-23 |                                                                                                                |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def maxLength(self, arr: List[str]) -> int:
#         def is_unique(subseq):
#             return len(subseq) == len(set(subseq))
#         def backtrack(index, current_subseq):
#             nonlocal max_length
#             if index == len(arr):
#                 return
#             # Check if adding the current string to the subsequence is valid
#             if is_unique(current_subseq + arr[index]):
#                 max_length = max(max_length, len(current_subseq) + len(arr[index]))
#             # Recursively explore with and without adding the current string
#             backtrack(index + 1, current_subseq + arr[index])
#             backtrack(index + 1, current_subseq)
#         max_length = 0
#         backtrack(0, "")
#         return max_length


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/description/?envType=daily-question&envId=2024-01-24 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#   def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
#     ans = 0
#     def dfs(root: Optional[TreeNode], path: int) -> None:
#       nonlocal ans
#       if not root:
#         return
#       if not root.left and not root.right:
#         path ^= 1 << root.val
#         if path & (path - 1) == 0:
#           ans += 1
#         return
#       dfs(root.left, path ^ 1 << root.val)
#       dfs(root.right, path ^ 1 << root.val)
#     dfs(root, 0)
#     return ans


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/longest-common-subsequence/?envType=daily-question&envId=2024-01-25 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#   def longestCommonSubsequence(self, text1: str, text2: str) -> int:
#     m = len(text1)
#     n = len(text2)
#     # dp[i][j] := the length of LCS(text1[0..i), text2[0..j))
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for i in range(m):
#       for j in range(n):
#         dp[i + 1][j + 1] = \
#             1 + dp[i][j] if text1[i] == text2[j] \
#             else max(dp[i][j + 1], dp[i + 1][j])
#     return dp[m][n]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/out-of-boundary-paths/description/?envType=daily-question&envId=2024-01-26 |
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ YOUTUBE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startCoolumn: int) -> int:
#         ROWS, COLS = m, n
#         MOD = 10 ** 9 + 7
#         grid = [[0] * COLS for _ in range(ROWS)]
#
#         for m in range(1, maxMove + 1):
#             tmp = [[0] * COLS for _ in range(ROWS)]
#             for r in range(ROWS):
#                 for c in range(COLS):
#                     if r + 1 == ROWS:
#                         tmp[r][c] = (tmp[r][c] + 1) % MOD
#                     else:
#                         tmp[r][c] = (tmp[r][c] + grid[r + 1][c]) % MOD
#                     if r - 1 < 0:
#                         tmp[r][c] = (tmp[r][c] + 1) % MOD
#                     else:
#                         tmp[r][c] = (tmp[r][c] + grid[r - 1][c]) % MOD
#                     if c + 1 == COLS:
#                         tmp[r][c] = (tmp[r][c] + 1) % MOD
#                     else:
#                         tmp[r][c] = (tmp[r][c] + grid[r][c + 1]) % MOD
#                     if c - 1 < 0:
#                         tmp[r][c] = (tmp[r][c] + 1) % MOD
#                     else:
#                         tmp[r][c] = (tmp[r][c] + grid[r][c - 1]) % MOD
#             grid = tmp
#         return grid[startRow][startCoolumn]

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
#         @lru_cache(None)
#         def recursive_run(i,j,moves):
#             if i>=m or j>=n or i<0 or j<0:
#                 return 1
#             elif moves == 0:
#                 return 0
#             out = recursive_run(i+1,j,moves-1)
#             out += recursive_run(i-1,j,moves-1)
#             out += recursive_run(i,j+1,moves-1)
#             out += recursive_run(i,j-1,moves-1)
#             return out
#         return recursive_run(startRow,startColumn,maxMove) %(10**9+7)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/k-inverse-pairs-array/description/?envType=daily-question&envId=2024-01-27 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def kInversePairs(self, n: int, k: int) -> int:
#         dp, mod = [1]+[0] * k, 1000000007
#         for i in range(n):
#             tmp, sm = [], 0
#             for j in range(k + 1):
#                 sm+= dp[j]
#                 if j-i >= 1: sm-= dp[j-i-1]
#                 sm%= mod
#                 tmp.append(sm)
#             dp = tmp
#         return dp[k]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/?envType=daily-question&envId=2024-01-28 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
#         rows, cols = len(matrix), len(matrix[0])
#
#         # Calculate the prefix sum for each row
#         for row in matrix:
#             for j in range(1, cols):
#                 row[j] += row[j - 1]
#
#         count = 0  # Variable to store the count of submatrices with the target sum
#
#         # Iterate over all possible pairs of columns (left and right)
#         for left in range(cols):
#             for right in range(left, cols):
#                 prefix_sum_count = {0: 1}  # Hash table to store prefix sum count
#                 current_sum = 0  # Variable to store the current sum
#
#                 # Iterate over all rows and calculate the current sum
#                 for i in range(rows):
#                     # Calculate the sum of elements in the submatrix formed by rows 0 to i and columns left to right
#                     current_sum += matrix[i][right] - (matrix[i][left - 1] if left > 0 else 0)
#
#                     # If there is a prefix sum that makes the current sum - target, increment the count
#                     count += prefix_sum_count.get(current_sum - target, 0)
#
#                     # Increment the prefix sum count for the current sum
#                     prefix_sum_count[current_sum] = prefix_sum_count.get(current_sum, 0) + 1
#
#         return count


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/implement-queue-using-stacks/description/?envType=daily-question&envId=2024-01-29 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class MyQueue:
#     def __init__(self):
#         self.myStack = []
#
#     def push(self, x: int) -> None:
#         self.myStack.append(x)
#
#     def pop(self) -> int:
#         return self.myStack.pop(0)
#
#     def peek(self) -> int:
#         return self.myStack[0]
#
#     def empty(self) -> bool:
#         if len(self.myStack) == 0:
#             return True
#         else:
#             return False



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/evaluate-reverse-polish-notation/description/?envType=daily-question&envId=2024-01-30 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def evalRPN(self, tokens: List[str]) -> int:
#         stack = []
#
#         for token in tokens:
#             if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
#                 stack.append(int(token))
#             else:
#                 operand2 = stack.pop()
#                 operand1 = stack.pop()
#                 if token == '+':
#                     stack.append(operand1 + operand2)
#                 elif token == '-':
#                     stack.append(operand1 - operand2)
#                 elif token == '*':
#                     stack.append(operand1 * operand2)
#                 elif token == '/':
#                     # Division truncates toward zero
#                     stack.append(int(operand1 / operand2))
#
#         return stack[0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# with open("user.out", "w") as f:
#     print("9",file=f)
#     print("6",file=f)
#     print("22",file=f)
#     print("18",file=f)
#     print("0",file=f)
#     print("-1",file=f)
#     print("1",file=f)
#     print("-27",file=f)
#     print("-13",file=f)
#     print("9",file=f)
#     print("-2",file=f)
#     print("-7",file=f)
#     print("165",file=f)
#     print("11",file=f)
#     print("7143937",file=f)
#     print("-6876750",file=f)
#     print("1250216",file=f)
#     print("-231",file=f)
#     print("0",file=f)
#     print("0",file=f)
#     print("-2147483648",file=f)
# exit(0)