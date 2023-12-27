# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/median-of-two-sorted-arrays/submissions/1129949277/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# # Example usage:
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