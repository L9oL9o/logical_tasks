# 1 MAY
# https://leetcode.com/problems/reverse-prefix-of-word/description/?envType=daily-question&envId=2024-05-01

# class Solution:
#     def reversePrefix(self, word: str, ch: str) -> str:
#         j = word.find(ch)
#         if j != -1:
#             return word[:j+1][::-1] + word[j+1:]
#         return word


# 2 MAY
# https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative/description/?envType=daily-question&envId=2024-05-02

# class Solution:
#     def findMaxK(self, nums: List[int]) -> int:
#         nums.sort()
#         n = len(nums)
#         for i in range(n-1, -1, -1):
#             if nums[i] > 0 and -nums[i] in nums:
#                 return nums[i]
#         return -1  # If no such pair found


# 3 MAY
# https://leetcode.com/problems/compare-version-numbers/description/?envType=daily-question&envId=2024-05-03

# class Solution:
#     def compareVersion(self, version1: str, version2: str) -> int:
#         def helper(s: str, idx: int) -> List[int]:
#             num = 0
#             while idx < len(s):
#                 if s[idx] == '.':
#                     break
#                 else:
#                     num = num * 10 + int(s[idx])
#                 idx += 1
#             return [num, idx+1]
#
#         i = j = 0
#         while(i < len(version1) or j < len(version2)):
#             v1, i = helper(version1, i)
#             v2, j = helper(version2, j)
#             if v1 > v2:
#                 return 1
#             elif v1 < v2:
#                 return -1
#
#         return 0


# 4 MAY
#


# 5 MAY
#


# 6 MAY
#


# 7 MAY
#


# 8 MAY
#


# 9 MAY
#


# 10 MAY
#


# 11 MAY
#


# 12 MAY
#


# 13 MAY
#


# 14 MAY
#


# 15 MAY
#


# 16 MAY
#


# 17 MAY
#


# 18 MAY
#


# 19 MAY
#


# 20 MAY
#


# 21 MAY
#


# 22 MAY
#


# 23 MAY
#


# 24 MAY
#


# 25 MAY
#


# 26 MAY
#


# 27 MAY
#


# 28 MAY
#


# 29 MAY
#


# 30 MAY
#


# 31 MAY
#
