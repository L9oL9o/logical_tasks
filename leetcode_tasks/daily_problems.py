# 3 January
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