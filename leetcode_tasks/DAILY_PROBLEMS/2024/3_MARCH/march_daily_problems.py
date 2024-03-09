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

# 11 MARCH

# 12 MARCH

# 13 MARCH

# 14 MARCH

# 15 MARCH

# 16 MARCH

# 17 MARCH

# 18 MARCH

# 19 MARCH

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
