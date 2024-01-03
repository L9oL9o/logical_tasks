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
