# https://leetcode.com/problems/two-sum/description/

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
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
