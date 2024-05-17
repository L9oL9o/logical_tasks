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
#https://leetcode.com/problems/boats-to-save-people/?envType=daily-question&envId=2024-05-04

# class Solution:
#     def numRescueBoats(self, p: List[int], limit: int) -> int:
#         p.sort()
#         x = 0
#         l, r = 0, len(p) - 1
#         while l <= r:
#             x += 1
#             if p[l] + p[r] <= limit:
#                 l += 1
#             r -= 1
#         return x


# 5 MAY
# https://leetcode.com/problems/delete-node-in-a-linked-list/description/?envType=daily-question&envId=2024-05-05

# class Solution:
#     def deleteNode(self, node: ListNode) -> None:
#         node.val = node.next.val
#         node.next = node.next.next


# 6 MAY
# https://leetcode.com/problems/remove-nodes-from-linked-list/description/?envType=daily-question&envId=2024-05-06

# class Solution:
#     def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         stack = []
#         current = head
#
#         while current:
#             while stack and stack[-1].val < current.val:
#                 stack.pop()
#
#             stack.append(current)
#             current = current.next
#
#         dummy = ListNode(0)
#         prev = dummy
#
#         for node in stack:
#             prev.next = node
#             prev = prev.next
#
#         prev.next = None
#
#         return dummy.next


# 7 MAY
# https://leetcode.com/problems/double-a-number-represented-as-a-linked-list/submissions/1251923770/?envType=daily-question&envId=2024-05-07

# class Solution:
#     def doubleIt(self, head: ListNode) -> ListNode:
#         # Reverse the linked list
#         reversed_list = self.reverse_list(head)
#         # Initialize variables to track carry and previous node
#         carry = 0
#         current, previous = reversed_list, None
#
#         # Traverse the reversed linked list
#         while current:
#             # Calculate the new value for the current node
#             new_value = current.val * 2 + carry
#             # Update the current node's value
#             current.val = new_value % 10
#             # Update carry for the next iteration
#             carry = 1 if new_value > 9 else 0
#             # Move to the next node
#             previous, current = current, current.next
#
#         # If there's a carry after the loop, add an extra node
#         if carry:
#             previous.next = ListNode(carry)
#
#         # Reverse the list again to get the original order
#         result = self.reverse_list(reversed_list)
#
#         return result
#
#     # Method to reverse the linked list
#     def reverse_list(self, node: ListNode) -> ListNode:
#         previous, current = None, node
#
#         # Traverse the original linked list
#         while current:
#             # Store the next node
#             next_node = current.next
#             # Reverse the link
#             current.next = previous
#             # Move to the next nodes
#             previous, current = current, next_node
#
#         # Previous becomes the new head of the reversed list
#         return previous


# 8 MAY
# https://leetcode.com/problems/relative-ranks/description/?envType=daily-question&envId=2024-05-08

# class Solution:
#     def findRelativeRanks(self, score: List[int]) -> List[str]:
#         N = len(score)
#
#         # Create a heap of pairs (score, index)
#         heap = []
#         for index, score in enumerate(score):
#             heapq.heappush(heap, (-score, index))
#
#         # Assign ranks to athletes
#         rank = [0] * N
#         place = 1
#         while heap:
#             original_index = heapq.heappop(heap)[1]
#             if place == 1:
#                 rank[original_index] = "Gold Medal"
#             elif place == 2:
#                 rank[original_index] = "Silver Medal"
#             elif place == 3:
#                 rank[original_index] = "Bronze Medal"
#             else:
#                 rank[original_index] = str(place)
#             place += 1
#
#         return rank




# 9 MAY
# https://leetcode.com/problems/maximize-happiness-of-selected-children/description/?envType=daily-question&envId=2024-05-09

# class Solution:
#     def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
#         happiness.sort(reverse=True)
#         i = 0
#         res = 0
#
#         while k > 0:
#             happiness[i] = max(happiness[i] - i, 0)
#             res += happiness[i]
#             i += 1
#             k -= 1
#
#         return res


# 10 MAY
# https://leetcode.com/problems/k-th-smallest-prime-fraction/description/?envType=daily-question&envId=2024-05-10

# class Solution:
#     def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
#         n = len(arr)
#         left, right = 0, 1
#         res = []
#
#         while left <= right:
#             mid = left + (right - left) / 2
#             j, total, num, den = 1, 0, 0, 0
#             maxFrac = 0
#             for i in range(n):
#                 while j < n and arr[i] >= arr[j] * mid:
#                     j += 1
#
#                 total += n - j
#
#                 if j < n and maxFrac < arr[i] * 1.0 / arr[j]:
#                     maxFrac = arr[i] * 1.0 / arr[j]
#                     num, den = i, j
#
#             if total == k:
#                 res = [arr[num], arr[den]]
#                 break
#
#             if total > k:
#                 right = mid
#             else:
#                 left = mid
#
#         return res


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
# https://leetcode.com/problems/evaluate-boolean-binary-tree/description/?envType=daily-question&envId=2024-05-16

# class Solution:
#     def helper(self, root):
#         if root.val == 0 or root.val == 1:
#             return root.val == 1
#         elif root.val == 2:
#             return self.helper(root.left) or self.helper(root.right)
#         elif root.val == 3:
#             return self.helper(root.left) and self.helper(root.right)
#         return False
#
#     def evaluateTree(self, root: Optional[TreeNode]) -> bool:
#         return self.helper(root)


# 17 MAY
#  https://leetcode.com/problems/delete-leaves-with-a-given-value/?envType=daily-question&envId=2024-05-17

# class Solution(object):
#     def removeLeafNodes(self, root, target):
#         """
#         :type root: TreeNode
#         :type target: int
#         :rtype: TreeNode
#         """
#         if not root:
#             return None
#         root.left = self.removeLeafNodes(root.left, target)
#         root.right = self.removeLeafNodes(root.right, target)
#         if not root.left and not root.right and root.val == target:
#             return None
#         return root


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
