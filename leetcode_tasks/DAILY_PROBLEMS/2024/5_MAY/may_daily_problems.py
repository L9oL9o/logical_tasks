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
#


# 9 MAY
#


# 10 MAY
#


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
#


# 17 MAY
#


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
