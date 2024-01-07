# 21 DECEMBER 2023
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/widest-vertical-area-between-two-points-containing-no-points/description/?envType=daily-question&envId=2023-12-21 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
#         points.sort(key=lambda x: x[0])  # Sort points based on x-coordinate
#
#         max_width = 0
#         for i in range(1, len(points)):
#             max_width = max(max_width, points[i][0] - points[i - 1][0])
#
#         return max_width


# 1 JANUARY 2024
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


# 2 JANUARY 2024
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



# 3 JANUARY 2024
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



# 4 JANUARY 2024
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



# 5 JANUARY 2024
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



# 6 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/maximum-profit-in-job-scheduling/?envType=daily-question&envId=2024-01-06 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
#         jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])  # Sort jobs by end time
#         n = len(jobs)
#
#         dp = [0] * n  # dp[i] represents the maximum profit achievable until job i
#
#         for i in range(n):
#             # Use binary search to find the latest non-overlapping job
#             prev_job_idx = self.binarySearch(jobs, i)
#
#             # Calculate the maximum profit for the current job
#             include_current = dp[prev_job_idx] + jobs[i][2]
#             exclude_current = dp[i - 1] if i > 0 else 0
#             dp[i] = max(include_current, exclude_current)
#
#         return dp[-1]
#
#     def binarySearch(self, jobs, current_idx):
#         low, high = 0, current_idx - 1
#
#         while low <= high:
#             mid = (low + high) // 2
#             if jobs[mid][1] <= jobs[current_idx][0]:
#                 if jobs[mid + 1][1] <= jobs[current_idx][0]:
#                     low = mid + 1
#                 else:
#                     return mid
#             else:
#                 high = mid - 1
#
#         return -1



# 7 JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/?envType=daily-question&envId=2024-01-07 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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