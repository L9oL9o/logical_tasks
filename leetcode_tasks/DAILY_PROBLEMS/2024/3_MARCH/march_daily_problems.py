# 01 MARCH
# https://leetcode.com/problems/maximum-odd-binary-number/description/?envType=daily-question&envId=2024-03-01

from builtins import list
# class Solution:
#     def maximumOddBinaryNumber(self, s: str) -> str:
#         ones = s.count('1')
#         return ('1' * (ones - 1)) + ('0' * (len(s) - ones)) + '1'


# 02 MARCH
# https://leetcode.com/problems/squares-of-a-sorted-array/description/?envType=daily-question&envId=2024-03-02
# ***************
# class Solution:
#     def sortedSquares(self, nums: List[int]) -> List[int]:
#         numes = [i ** 2 for i in nums]
#         return numes.sort()


# 03 MARCH
# https://leetcode.com/problems/remove-nth-node-from-end-of-list/?envType=daily-question&envId=2024-03-03

# class Solution:
#     def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
#         dummy = ListNode(0)
#         dummy.next = head
#         first = dummy
#         second = dummy
#         for _ in range(n + 1):
#             first = first.next
#         while first is not None:
#             first = first.next
#             second = second.next
#         second.next = second.next.next
#         return dummy.next


# 04 MARCH
# https://leetcode.com/problems/bag-of-tokens/description/?envType=daily-question&envId=2024-03-04

# class Solution:
#   def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
#     ans = 0
#     score = 0
#     q = collections.deque(sorted(tokens))
#     while q and (power >= q[0] or score):
#       while q and power >= q[0]:
#         power -= q.popleft()
#         score += 1
#       ans = max(ans, score)
#       if q and score:
#         power += q.pop()
#         score -= 1
#     return ans


# 05 MARCH
# https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends/description/?envType=daily-question&envId=2024-03-05

# class Solution:
#     def minimumLength(self, s: str) -> int:
#         l, r = 0, len(s) - 1
#         while l < r and s[l] == s[r]:
#             char = s[l]
#             l += 1
#             r -= 1
#             while l <= r and s[l] == char:
#                 l += 1
#             while l <= r and s[r] == char:
#                 r -= 1
#         return r - l + 1


# 06 MARCH
# https://leetcode.com/problems/linked-list-cycle/description/?envType=daily-question&envId=2024-03-06

# class Solution:
#     def hasCycle(self, head: Optional[ListNode]) -> bool:
#         fast = slow = head
#         while fast and fast.next:
#             slow, fast = slow.next, fast.next.next
#             if fast == slow:
#                 return True
#         return False


# 07 MARCH
# https://leetcode.com/problems/middle-of-the-linked-list/?envType=daily-question&envId=2024-03-07

# class Solution:
#     def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         slow_pointer = head
#         fast_pointer = head
#         while fast_pointer is not None and fast_pointer.next is not None:
#             slow_pointer = slow_pointer.next
#             fast_pointer = fast_pointer.next.next
#         return slow_pointer


# 08 MARCH
# https://leetcode.com/problems/count-elements-with-maximum-frequency/description/?envType=daily-question&envId=2024-03-08

# class Solution:
#     def maxFrequencyElements(self, nums: List[int]) -> int:
#         from collections import defaultdict
#
#         counts = defaultdict(int)
#         max_count = 0
#
#         for num in nums:
#             counts[num] += 1
#             max_count = max(max_count, counts[num])
#
#         ans = 0
#         for v in counts.values():
#             if v == max_count:
#                 ans += v
#
#         return ans


# 09 MARCH
# https://leetcode.com/problems/minimum-common-value/description/?envType=daily-question&envId=2024-03-09

# class Solution:
#     def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
#         n1, n2 = len(nums1), len(nums2)
#         p1, p2 = 0, 0
#         while p1 < n1 and p2 < n2:
#             x = nums1[p1]
#             y = nums2[p2]
#             if x == y:
#                 return x
#             elif x > y:
#                 p2 += 1
#             else:
#                 p1 += 1
#         return -1


# 10 MARCH
# https://leetcode.com/problems/intersection-of-two-arrays/description/?envType=daily-question&envId=2024-03-10

# class Solution:
#     def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
#         set1 = set(nums1)
#         set2 = set(nums2)
#         return list(set1.intersection(set2))


# 11 MARCH
# https://leetcode.com/problems/custom-sort-string/description/?envType=daily-question&envId=2024-03-11

# class Solution:
#     def customSortString(self, order: str, s: str) -> str:
#         ctr = Counter(s)
#         ans = [ch * ctr[ch] for ch in order]
#         ans.extend(filter(lambda x: x not in order, s))
#         return ''.join(ans)


# 12 MARCH
# https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/submissions/1201655509/?envType=daily-question&envId=2024-03-12

# class Solution:
#     def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         dummy = ListNode(0)
#         dummy.next = head
#         prefix_sum = 0
#         prefix_sums = {0: dummy}
#         current = head
#         while current:
#             prefix_sum += current.val
#             if prefix_sum in prefix_sums:
#                 to_delete = prefix_sums[prefix_sum].next
#                 temp_sum = prefix_sum + to_delete.val
#                 while to_delete != current:
#                     del prefix_sums[temp_sum]
#                     to_delete = to_delete.next
#                     temp_sum += to_delete.val
#                 prefix_sums[prefix_sum].next = current.next
#             else:
#                 prefix_sums[prefix_sum] = current
#             current = current.next
#
#         return dummy.next


# 13 MARCH
# https://leetcode.com/problems/find-the-pivot-integer/description/?envType=daily-question&envId=2024-03-13

# class Solution:
#     def pivotInteger(self, n: int) -> int:
#         x = sqrt(n * (n + 1) / 2)
#         if x % 1 != 0:
#             return -1
#         else:
#             return int(x)


# 14 MARCH
# https://leetcode.com/problems/binary-subarrays-with-sum/description/?envType=daily-question&envId=2024-03-14

# class Solution:
#     def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
#         zeros, prev = [], -1
#         for i, num in enumerate(nums+[1]):
#             if num:
#                 zeros.append(i-prev)
#                 prev = i
#         return sum(cnt*zeros[i] for i, cnt in enumerate(zeros[goal:])) if goal else sum(cnt*(cnt-1)//2 for cnt in zeros)


# 15 MARCH
# https://leetcode.com/problems/product-of-array-except-self/?envType=daily-question&envId=2024-03-15

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


# 16 MARCH
# https://leetcode.com/problems/contiguous-array/description/?envType=daily-question&envId=2024-03-16

# sys.stdout = open('user.out', 'w')
# for nums in map(loads, stdin):
#     totalsum,hashmap=0,{0:-1}
#     res,diff=0,0
#     for i in range(len(nums)):
#         if(nums[i]==0):
#             totalsum-=1
#         else:
#             totalsum+=1
#         try:
#             diff=i-hashmap[totalsum]
#             if(diff>res):
#                 res=diff
#         except:
#             hashmap[totalsum]=i
#     print(res)


# 17 MARCH
# https://leetcode.com/problems/insert-interval/description/?envType=daily-question&envId=2024-03-17

# class Solution:
#     def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
#         merged = []
#         i = 0
#
#         while i < len(intervals) and intervals[i][1] < newInterval[0]:
#             merged.append(intervals[i])
#             i += 1
#
#         while i < len(intervals) and intervals[i][0] <= newInterval[1]:
#             newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
#             i += 1
#         merged.append(newInterval)
#
#         while i < len(intervals):
#             merged.append(intervals[i])
#             i += 1
#
#         return merged


# 18 MARCH
# https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/?envType=daily-question&envId=2024-03-18

# class Solution:
#     def findMinArrowShots(self, points: List[List[int]]) -> int:
#         n=len(points)
#         points = sorted(points, key = lambda x: x[1])
#         maxa=-float('inf')
#         ans=0
#         for i in range(0,n):
#             if maxa<points[i][0]:
#                 ans+=1
#                 maxa=points[i][1]
#         return ans


# 19 MARCH
# https://leetcode.com/problems/task-scheduler/description/?envType=daily-question&envId=2024-03-19

# class Solution:
#     def leastInterval(self, tasks: List[str], n: int) -> int:
#         freq = [0] * 26
#         for task in tasks:
#             freq[ord(task) - ord('A')] += 1
#         freq.sort()
#         chunk = freq[25] - 1
#         idle = chunk * n
#         for i in range(24, -1, -1):
#             idle -= min(chunk, freq[i])
#         return len(tasks) + idle if idle >= 0 else len(tasks)


# 20 MARCH

# 21 MARCH

# 22 MARCH

# 23 MARCH

# 24 MARCH

# 25 MARCH

# 26 MARCH

# 27 MARCH

# 28 MARCH

# 29 MARCH

# 30 MARCH

# 31 MARCH
