# https://leetcode.com/problems/merge-strings-alternately/description/?envType=study-plan-v2&envId=leetcode-75
# 1
# class Solution:
#     def mergeAlternately(self, word1: str, word2: str) -> str:
#         new_str = ""
#         ind = 0
#         while ind < len(word1) or ind < len(word2):
#             char1 = word1[ind] if ind < len(word1) else ""
#             char2 = word2[ind] if ind < len(word2) else ""
#             new_str += char1 + char2
#             ind += 1
#         return new_str


# https://leetcode.com/problems/greatest-common-divisor-of-strings/description/?envType=study-plan-v2&envId=leetcode-75
# 2
# class Solution:
#     from math import gcd
#     def gcdOfStrings(self, str1: str, str2: str) -> str:
#         if str1 + str2 != str2 + str1:
#             return ""
#         return str1[:gcd(len(str1), len(str2))]
#


# https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/?envType=study-plan-v2&envId=leetcode-75
# 3 ***
# class Solution:
#     def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
#         result = []
#         for i in candies:
#             if (i + extraCandies) >= max(candies):
#                 result.append(True)
#             else:
#                 result.append(False)
#         return result

# https://leetcode.com/problems/can-place-flowers/description/?envType=study-plan-v2&envId=leetcode-75
# 4
# class Solution:
#     def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
#         flowerbed.insert(0, 0)
#         flowerbed.append(0)
#         for i in range(1, len(flowerbed) - 1):
#             if n == 0:
#                 return True
#             else:
#                 if flowerbed[i] + flowerbed[i + 1] == 0 and flowerbed[i - 1] != 1:
#                     flowerbed[i] = 1
#                     n -= 1
#         return n <= 0


# https://leetcode.com/problems/reverse-vowels-of-a-string/description/?envType=study-plan-v2&envId=leetcode-75
# 5
# class Solution:
#     def reverseVowels(self, s: str) -> str:
#         s = list(s) # convert string to list
#         vowels = set("aeiouAEIOU")
#         low = 0
#         high = len(s) - 1
#         while low < high:
#             if s[low] not in vowels:
#                 low += 1
#             elif s[high] not in vowels:
#                 high -= 1
#             else:
#                 s[low], s[high] = s[high], s[low] # swap the chars
#                 low += 1
#                 high -= 1
#         return "".join(s) # convert list to string


# https://leetcode.com/problems/reverse-words-in-a-string/description/?envType=study-plan-v2&envId=leetcode-75
# 6
# class Solution:
#     def reverseWords(self, s: str) -> str:
#         s = s.split()
#         return " ".join(s[::-1])


# https://leetcode.com/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=leetcode-75
# 7
# class Solution:
#     def productExceptSelf(self, nums: List[int]) -> List[int]:
#         n = len(nums)
#         prefix_product = 1
#         postfix_product = 1
#         result = [0]*n
#         for i in range(n):
#             result[i] = prefix_product
#             prefix_product *= nums[i]
#         for i in range(n-1,-1,-1):
#             result[i] *= postfix_product
#             postfix_product *= nums[i]
#         return result

# https://leetcode.com/problems/increasing-triplet-subsequence/?envType=study-plan-v2&envId=leetcode-75
# 8
# class Solution:
#     def increasingTriplet(self, nums: List[int]) -> bool:
#         first = second = float('inf')
#         for n in nums:
#             if n <= first:
#                 first = n
#             elif n <= second:
#                 second = n
#             else:
#                 return True
#         return False


# https://leetcode.com/problems/string-compression/description/?envType=study-plan-v2&envId=leetcode-75
# 9
# class Solution:
#   def compress(self, chars: List[str]) -> int:
#     ans = 0
#     i = 0
#     while i < len(chars):
#       letter = chars[i]
#       count = 0
#       while i < len(chars) and chars[i] == letter:
#         count += 1
#         i += 1
#       chars[ans] = letter
#       ans += 1
#       if count > 1:
#         for c in str(count):
#           chars[ans] = c
#           ans += 1
#     return ans


# https://leetcode.com/problems/move-zeroes/description/?envType=study-plan-v2&envId=leetcode-75
# 10
# class Solution:
#     def moveZeroes(self, nums: list) -> None:
#         slow = 0
#         for fast in range(len(nums)):
#             if nums[fast] != 0 and nums[slow] == 0:
#                 nums[slow], nums[fast] = nums[fast], nums[slow]
#             # wait while we find a non-zero element to
#             # swap with you
#             if nums[slow] != 0:
#                 slow += 1


# https://leetcode.com/problems/is-subsequence/description/?envType=study-plan-v2&envId=leetcode-75
# 11
# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         if s == '':
#             return True
#         index = 0
#         # iterate through 's', add pointer to 't'
#         for i in t:
#             if i == s[index]:
#                 index += 1
#                 if index == len(s):
#                     return True
#         return False


# https://leetcode.com/problems/container-with-most-water/description/?envType=study-plan-v2&envId=leetcode-75
# 12
# f = open('user.out', 'w')
# for height in map(loads, stdin):
#     left = 0
#     right = len(height) - 1
#     maxWater = 0
#     maxh = max(height)
#     while left < right:
#         currWater = min(height[left], height[right]) * (right - left)
#         maxWater = max(maxWater, currWater)
#         if maxWater >= maxh * (right-left):
#             break
#         if height[left] < height[right]:
#             left += 1
#         else:
#             right -= 1
#     print(maxWater, file=f)
# exit(0)
