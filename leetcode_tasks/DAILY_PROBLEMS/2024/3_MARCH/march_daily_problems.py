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
# https://leetcode.com/problems/merge-in-between-linked-lists/description/?envType=daily-question&envId=2024-03-20

# class Solution:
#     def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
#         ptr = list1
#         for _ in range(a - 1):
#             ptr = ptr.next
#
#         qtr = ptr.next
#         for _ in range(b - a + 1):
#             qtr = qtr.next
#
#         ptr.next = list2
#         while list2.next:
#             list2 = list2.next
#         list2.next = qtr
#
#         return list1


# 21 MARCH
# https://leetcode.com/problems/reverse-linked-list/description/?envType=daily-question&envId=2024-03-21

# class Solution:
#     def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         prev_node = None
#         current_node = head
#
#         while current_node is not None:
#             next_node = current_node.next
#             current_node.next = prev_node
#             prev_node = current_node
#             current_node = next_node
#
#         return prev_node


# 22 MARCH
# https://leetcode.com/problems/palindrome-linked-list/description/?envType=daily-question&envId=2024-03-22

# class Solution:
#     def isPalindrome(self, head: Optional[ListNode]) -> bool:
#         list_vals = []
#         while head:
#             list_vals.append(head.val)
#             head = head.next
#
#         left, right = 0, len(list_vals) - 1
#         while left < right and list_vals[left] == list_vals[right]:
#             left += 1
#             right -= 1
#         return left >= right


# 23 MARCH
# https://leetcode.com/problems/reorder-list/description/?envType=daily-question&envId=2024-03-23

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# class Solution:
#     def reorderList(self, head: Optional[ListNode]) -> None:
#         """
#         Do not return anything, modify head in-place instead.
#         """
#         if not head or not head.next:
#             return
#
#         # Step 1: Find the middle of the linked list
#         slow, fast = head, head
#         while fast.next and fast.next.next:
#             slow = slow.next
#             fast = fast.next.next
#
#         # Split the linked list into two halves
#         second_half = slow.next
#         slow.next = None
#
#         # Step 2: Reverse the second half of the linked list
#         prev = None
#         current = second_half
#         while current:
#             next_node = current.next
#             current.next = prev
#             prev = current
#             current = next_node
#         second_half = prev
#
#         # Step 3: Merge the first half and the reversed second half alternately
#         first_half = head
#         while second_half:
#             next_first = first_half.next
#             next_second = second_half.next
#
#             first_half.next = second_half
#             second_half.next = next_first
#
#             first_half = next_first
#             second_half = next_second


# 24 MARCH
# https://leetcode.com/problems/find-the-duplicate-number/description/?envType=daily-question&envId=2024-03-24

# class Solution {
#     public int findDuplicate(int[] nums) {
#         for(int i=0;i<nums.length;i++)
#             for(int j=i+1;j<nums.length;j++)
#                 if(nums[i] == nums[j]) return nums[i];
#         return -1;
#     }
# }


# 25 MARCH
# https://leetcode.com/problems/find-all-duplicates-in-an-array/description/?envType=daily-question&envId=2024-03-25

# class Solution:
#     def findDuplicates(self, nums: List[int]) -> List[int]:
#         ans =[]
#         n=len(nums)
#         for x in nums:
#             x = abs(x)
#             if nums[x-1]<0:
#                 ans.append(x)
#             nums[x-1] *= -1
#         return ans


# 26 MARCH
# https://leetcode.com/problems/first-missing-positive/description/?envType=daily-question&envId=2024-03-26

# class Solution:
#     def firstMissingPositive(self, nums: List[int]) -> int:
#         # Function to swap elements in the array
#         def swap(arr, i, j):
#             arr[i], arr[j] = arr[j], arr[i]
#
#         n = len(nums)
#
#         # Place each positive integer i at index i-1 if possible
#         for i in range(n):
#             while 0 < nums[i] <= n and nums[i] != nums[nums[i] - 1]:
#                 swap(nums, i, nums[i] - 1)
#
#         # Find the first missing positive integer
#         for i in range(n):
#             if nums[i] != i + 1:
#                 return i + 1
#
#         # If all positive integers from 1 to n are present, return n + 1
#         return n + 1


# 27 MARCH
# https://leetcode.com/problems/subarray-product-less-than-k/description/?envType=daily-question&envId=2024-03-27

# class Solution:
#     def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
#         if k <= 1:  # If k <= 1, no subarray can satisfy the condition
#             return 0
#
#         count = 0
#         prod = 1  # Product of elements in the current window
#         left = 0  # Left pointer of the window
#
#         for right, num in enumerate(nums):
#             prod *= num  # Expand the window by multiplying the current number
#             while prod >= k:  # Shrink the window from the left until the product is less than k
#                 prod /= nums[left]
#                 left += 1
#             count += right - left + 1  # Add the number of valid subarrays ending at the current position
#
#         return count

# 28 MARCH
# https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description/?envType=daily-question&envId=2024-03-28

# class Solution:
#     def maxSubarrayLength(self, nums: List[int], k: int) -> int:
#         ans = 0
#         mp = {}
#         l = 0
#         n = len(nums)
#         for r in range(n):
#             mp[nums[r]] = mp.get(nums[r], 0) + 1
#             if mp[nums[r]] > k:
#                 while nums[l] != nums[r]:
#                     mp[nums[l]] -= 1
#                     l += 1
#                 mp[nums[l]] -= 1
#                 l += 1
#             ans = max(ans, r - l + 1)
#         return ans


# 29 MARCH
# https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/description/?envType=daily-question&envId=2024-03-29

# class Solution:
#     def countSubarrays(self, nums: List[int], k: int) -> int:
#         max_val = max(nums)
#         result = start = max_count_in_window = 0
#
#         for end in range(len(nums)):
#             if nums[end] == max_val:
#                 max_count_in_window += 1
#             while max_count_in_window == k:
#                 if nums[start] == max_val:
#                     max_count_in_window -= 1
#                 start += 1
#             result += start
#         return result


# 30 MARCH
# https://leetcode.com/problems/subarrays-with-k-different-integers/description/?envType=daily-question&envId=2024-03-30

# from collections import defaultdict
#
# class Solution:
#     def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
#         def atMostK(nums, k):
#             counter = defaultdict(int)
#             distinct = 0
#             left = 0
#             result = 0
#
#             for right in range(len(nums)):
#                 if counter[nums[right]] == 0:
#                     distinct += 1
#                 counter[nums[right]] += 1
#
#                 while distinct > k:
#                     counter[nums[left]] -= 1
#                     if counter[nums[left]] == 0:
#                         distinct -= 1
#                     left += 1
#
#                 result += right - left + 1
#
#             return result
#
#         return atMostK(nums, k) - atMostK(nums, k - 1)


# 31 MARCH
# https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description/?envType=daily-question&envId=2024-03-31

# class Solution:
#     def countSubarrays(self, nums: List[int], mink: int, maxK: int) -> int:
#
#         res = 0
#         bad_idx = left_idx = right_idx = -1
#
#         for i, num in enumerate(nums) :
#             if not mink <= num <= maxK:
#                 bad_idx = i
#
#             if num == mink:
#                 left_idx = i
#
#             if num == maxK:
#                 right_idx = i
#
#             res += max(0, min(left_idx, right_idx) - bad_idx)
#
#         return res