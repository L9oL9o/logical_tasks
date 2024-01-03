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