# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        sorted_list = list(sorted(head))
        return sorted_list



class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Convert linked list values to a list
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next

        # Sort the list
        sorted_values = sorted(values)

        # Create a new linked list with sorted values
        dummy = ListNode()
        current = dummy
        for value in sorted_values:
            current.next = ListNode(value)
            current = current.next

        return dummy.next
