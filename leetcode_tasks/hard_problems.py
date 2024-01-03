# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/median-of-two-sorted-arrays/description/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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



# ~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/regular-expression-matching/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         # Create a 2D DP array to store matching results
#         dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
#
#         # Empty string and empty pattern match
#         dp[0][0] = True
#
#         # Handle patterns with '*'
#         for j in range(1, len(p) + 1):
#             if p[j - 1] == '*':
#                 dp[0][j] = dp[0][j - 2]
#
#         # Fill in the DP array
#         for i in range(1, len(s) + 1):
#             for j in range(1, len(p) + 1):
#                 if p[j - 1] == s[i - 1] or p[j - 1] == '.':
#                     dp[i][j] = dp[i - 1][j - 1]
#                 elif p[j - 1] == '*':
#                     dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] if s[i - 1] == p[j - 2] or p[j - 2] == '.' else False)
#
#         return dp[-1][-1]



# ~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/merge-k-sorted-lists |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# from queue import PriorityQueue
# from typing import List, Optional
#
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
# class Solution:
#     def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
#         # Create a priority queue (min-heap) to keep track of the current minimum node
#         min_heap = PriorityQueue()
#
#         # Add the first node from each list to the min-heap
#         for i, lst in enumerate(lists):
#             if lst:
#                 min_heap.put((lst.val, i, lst))
#
#         # Dummy node to simplify the code
#         dummy = ListNode()
#         current = dummy
#
#         while not min_heap.empty():
#             val, index, node = min_heap.get()
#             current.next = node
#             current = current.next
#
#             # Move to the next node in the list
#             if node.next:
#                 min_heap.put((node.next.val, index, node.next))
#
#         return dummy.next



# ~~~~~~~~~~~~~~~~~~~~~~~ HARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/substring-with-concatenation-of-all-words/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def findSubstring(self, s: str, words: List[str]) -> List[int]:
#         if not s or not words:
#             return []
#
#         word_len = len(words[0])
#         word_count = len(words)
#         total_len = word_len * word_count
#         word_freq = Counter(words)
#
#         result = []
#
#         for i in range(word_len):
#             left, right = i, i
#             current_window = Counter()
#
#             while right + word_len <= len(s):
#                 current_word = s[right:right + word_len]
#                 right += word_len
#                 current_window[current_word] += 1
#
#                 while current_window[current_word] > word_freq[current_word]:
#                     current_window[s[left:left + word_len]] -= 1
#                     left += word_len
#
#                 if right - left == total_len:
#                     result.append(left)
#
#         return result