# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # # https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/?envType=daily-question&envId=2023-l2-27 |
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # ~~~~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# from typing import List
#
#
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


class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        l = res = 0
        for r in range(1, len(colors)):
            if colors[l] == colors[r]:
                if neededTime[l] < neededTime[r]:
                    res += neededTime[l]
                    l = r
                else:
                    res += neededTime[r]
            else:
                l = r
        return res
