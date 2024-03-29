# https://leetcode.com/problems/merge-strings-alternately/description/?envType=study-plan-v2&envId=leetcode-75
# 1 Merge Strings Alternately
# class Solution:
#     def mergeAlternately(self, word1: str, word2: str) -> str:
#         new_str = ""
#         ind = 0
#         while ind < len(word1) or ind < len(word2):
#             char1 = word1[ind] if ind < len(word1) else ""
#             char2 = word2[ind] if ind < len(word2) else ""
#             new_str += char1 + char2
#             ind += 1
#         return new_str


# https://leetcode.com/problems/greatest-common-divisor-of-strings/description/?envType=study-plan-v2&envId=leetcode-75
# 2 Greatest Common Divisor of Strings
# class Solution:
#     from math import gcd
#     def gcdOfStrings(self, str1: str, str2: str) -> str:
#         if str1 + str2 != str2 + str1:
#             return ""
#         return str1[:gcd(len(str1), len(str2))]
#


# https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/?envType=study-plan-v2&envId=leetcode-75
# 3 *** Kids With the Greatest Number of Candies
# class Solution:
#     def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
#         result = []
#         for i in candies:
#             if (i + extraCandies) >= max(candies):
#                 result.append(True)
#             else:
#                 result.append(False)
#         return result

# https://leetcode.com/problems/can-place-flowers/description/?envType=study-plan-v2&envId=leetcode-75
# 4 Can Place Flowers
# class Solution:
#     def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
#         flowerbed.insert(0, 0)
#         flowerbed.append(0)
#         for i in range(1, len(flowerbed) - 1):
#             if n == 0:
#                 return True
#             else:
#                 if flowerbed[i] + flowerbed[i + 1] == 0 and flowerbed[i - 1] != 1:
#                     flowerbed[i] = 1
#                     n -= 1
#         return n <= 0


# https://leetcode.com/problems/reverse-vowels-of-a-string/description/?envType=study-plan-v2&envId=leetcode-75
# 5 Reverse Vowels of a String
# class Solution:
#     def reverseVowels(self, s: str) -> str:
#         s = list(s) # convert string to list
#         vowels = set("aeiouAEIOU")
#         low = 0
#         high = len(s) - 1
#         while low < high:
#             if s[low] not in vowels:
#                 low += 1
#             elif s[high] not in vowels:
#                 high -= 1
#             else:
#                 s[low], s[high] = s[high], s[low] # swap the chars
#                 low += 1
#                 high -= 1
#         return "".join(s) # convert list to string


# https://leetcode.com/problems/reverse-words-in-a-string/description/?envType=study-plan-v2&envId=leetcode-75
# 6 Reverse Words in a String
# class Solution:
#     def reverseWords(self, s: str) -> str:
#         s = s.split()
#         return " ".join(s[::-1])


# https://leetcode.com/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=leetcode-75
# 7 Product of Array Except Self
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

# https://leetcode.com/problems/increasing-triplet-subsequence/?envType=study-plan-v2&envId=leetcode-75
# 8 Increasing Triplet Subsequence
# class Solution:
#     def increasingTriplet(self, nums: List[int]) -> bool:
#         first = second = float('inf')
#         for n in nums:
#             if n <= first:
#                 first = n
#             elif n <= second:
#                 second = n
#             else:
#                 return True
#         return False


# https://leetcode.com/problems/string-compression/description/?envType=study-plan-v2&envId=leetcode-75
# 9 String Compression
# class Solution:
#   def compress(self, chars: List[str]) -> int:
#     ans = 0
#     i = 0
#     while i < len(chars):
#       letter = chars[i]
#       count = 0
#       while i < len(chars) and chars[i] == letter:
#         count += 1
#         i += 1
#       chars[ans] = letter
#       ans += 1
#       if count > 1:
#         for c in str(count):
#           chars[ans] = c
#           ans += 1
#     return ans


# https://leetcode.com/problems/move-zeroes/description/?envType=study-plan-v2&envId=leetcode-75
# 10 Move Zeroes
# class Solution:
#     def moveZeroes(self, nums: list) -> None:
#         slow = 0
#         for fast in range(len(nums)):
#             if nums[fast] != 0 and nums[slow] == 0:
#                 nums[slow], nums[fast] = nums[fast], nums[slow]
#             # wait while we find a non-zero element to
#             # swap with you
#             if nums[slow] != 0:
#                 slow += 1


# https://leetcode.com/problems/is-subsequence/description/?envType=study-plan-v2&envId=leetcode-75
# 11 Is Subsequence
# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         if s == '':
#             return True
#         index = 0
#         # iterate through 's', add pointer to 't'
#         for i in t:
#             if i == s[index]:
#                 index += 1
#                 if index == len(s):
#                     return True
#         return False


# https://leetcode.com/problems/container-with-most-water/description/?envType=study-plan-v2&envId=leetcode-75
# 12 Container With Most Water
# f = open('user.out', 'w')
# for height in map(loads, stdin):
#     left = 0
#     right = len(height) - 1
#     maxWater = 0
#     maxh = max(height)
#     while left < right:
#         currWater = min(height[left], height[right]) * (right - left)
#         maxWater = max(maxWater, currWater)
#         if maxWater >= maxh * (right-left):
#             break
#         if height[left] < height[right]:
#             left += 1
#         else:
#             right -= 1
#     print(maxWater, file=f)
# exit(0)


# https://leetcode.com/problems/max-number-of-k-sum-pairs/description/?envType=study-plan-v2&envId=leetcode-75# 13 Max Number of K-Sum Pairs
# 13 Max Number of K-Sum Pairs
# class Solution:
#     def maxOperations(self, nums: List[int], k: int) -> int:
#         nums = sorted(nums)
#
#         left = 0
#         right = len(nums) - 1
#         answer = 0
#
#         while left < right:
#             if nums[left] + nums[right] > k:
#                 right -= 1
#             elif nums[left] + nums[right] < k:
#                 left += 1
#             else:
#                 answer += 1
#                 left += 1
#                 right -= 1
#
#         return answer


# https://leetcode.com/problems/maximum-average-subarray-i/description/?envType=study-plan-v2&envId=leetcode-75
# 14 Maximum Average Subarray I
# def findMaxAverage(nums, k):
#     slide = sum(nums[:k])
#     mx = slide
#     for b in range(0,len(nums)-k):
#         slide = slide - nums[b] + nums[k+b]
#         if slide > mx:
#             mx = slide
#     return str(mx/k)
#
# f = open('user.out','w')
# a = 0
# for case in map(loads, stdin):
#     a += 1
#     if a == 1:
#         arr = case
#     else:
#         f.write(findMaxAverage(arr,case) + "\n")
#         a = 0
# f.close()
# exit(0)


# https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/?envType=study-plan-v2&envId=leetcode-75
# 15 Maximum Number of Vowels in a Substring of Given Length
# class Solution:
#     def maxVowels(self, s: str, k: int) -> int:
#         vowels=set("aeiou")
#         count = maxCount=0
#
#         for i in s[0:k]:
#             if i in vowels:
#                 count += 1
#         maxCount = count
#
#         for i in range(k,len(s)):
#             if s[i-k] in vowels:
#                 count -= 1
#             if s[i] in vowels:
#                 count += 1
#             if maxCount < count:
#                 maxCount = count
#         return maxCount


# https://leetcode.com/problems/max-consecutive-ones-iii/description/?envType=study-plan-v2&envId=leetcode-75
# 16 Max Consecutive Ones III
# class Solution:
#     def longestOnes(self, nums: List[int], k: int) -> int:
#         l=r=0
#         for r in range(len(nums)):
#             if nums[r] == 0:
#                 k-=1
#             if k<0:
#                 if nums[l] == 0:
#                     k+=1
#                 l+=1
#         return r-l+1


# https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/?envType=study-plan-v2&envId=leetcode-75
# 17 Longest Subarray of 1's After Deleting One Element
# class Solution:
#     def longestSubarray(self, nums: List[int]) -> int:
#         n = len(nums)
#         left = 0
#         zeros = 0
#         ans = 0
#         for right in range(n):
#             if nums[right] == 0:
#                 zeros += 1
#             while zeros > 1:
#                 if nums[left] == 0:
#                     zeros -= 1
#                 left += 1
#             ans = max(ans, right - left + 1 - zeros)
#         return ans - 1 if ans == n else ans


# https://leetcode.com/problems/find-the-highest-altitude/description/?envType=study-plan-v2&envId=leetcode-75
# 18 Find the Highest Altitude
# class Solution:
#     def largestAltitude(self, gain: List[int]) -> int:
#         k,s=0,0
#         for i in gain:
#             k+=i
#             s=max(s,k)
#         return s


# https://leetcode.com/problems/find-pivot-index/description/?envType=study-plan-v2&envId=leetcode-75
# 19 Find Pivot Index
# class Solution(object):
#     def pivotIndex(self, nums):
#         leftSum, rightSum = 0, sum(nums)
#         for idx, ele in enumerate(nums):
#             rightSum -= ele
#             if leftSum == rightSum:
#                 return idx
#             leftSum += ele
#         return -1


# https://leetcode.com/problems/find-the-difference-of-two-arrays/description/?envType=study-plan-v2&envId=leetcode-75
# 20 Find the Difference of Two Arrays
# class Solution:
#     def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
#         set1 = set(nums1)
#         set2 = set(nums2)
#         ans = [[], []]
#         for num in set1:
#             if num not in set2:
#                 ans[0].append(num)
#         for num in set2:
#             if num not in set1:
#                 ans[1].append(num)
#         return ans


# https://leetcode.com/problems/unique-number-of-occurrences/description/?envType=study-plan-v2&envId=leetcode-75
# 21 Unique Number of Occurrences
# class Solution:
#     def uniqueOccurrences(self, arr: List[int]) -> bool:
#         memo = []
#         for i in Counter(arr).values():
#
#             if i in memo: return False
#             else:   memo.append(i)
#
#         return True


# https://leetcode.com/problems/determine-if-two-strings-are-close/description/?envType=study-plan-v2&envId=leetcode-75
# 22 Determine if Two Strings Are Close
# class Solution:
#     def closeStrings(self, word1: str, word2: str) -> bool:
#         return len(word1) == len(word2) and all(
#             (chr(x) in word1) == (chr(x) in word2) for x in range(ord('a'), ord('z') + 1)) and sorted(
#             [word1.count(chr(x)) for x in range(ord('a'), ord('z') + 1)]) == sorted(
#             [word2.count(chr(x)) for x in range(ord('a'), ord('z') + 1)])


# https://leetcode.com/problems/equal-row-and-column-pairs/description/?envType=study-plan-v2&envId=leetcode-75
# 23 Equal Row and Column Pairs
# class Solution:
#     def equalPairs(self, grid: List[List[int]]) -> int:
#         # Store the first row of the grid
#         column_starts_with = grid[0]
#         # Initialize a list to store the columns of the grid
#         columns = [[] for _ in column_starts_with]
#         # Iterate through each row of the grid
#         for row in grid:
#             # Iterate through each element in the row and append it to the corresponding column
#             for j, element in enumerate(row):
#                 columns[j].append(element)
#         # Initialize a variable to count the equal pairs
#         equal_pairs = 0
#         # Iterate through each row of the grid
#         for row in grid:
#             # Iterate through each element in the first row
#             for j, element in enumerate(column_starts_with):
#                 # Check if the first element of the row is equal to the element in the corresponding column
#                 if row[0] == element:
#                     # Check if the entire row is equal to the column
#                     if row == columns[j]:
#                         equal_pairs += 1
#         return equal_pairs


# https://leetcode.com/problems/removing-stars-from-a-string/description/?envType=study-plan-v2&envId=leetcode-75
# 24 Removing Stars From a String
# class Solution:
#     def removeStars(self, s: str) -> str:
#         stack = []
#         for i in s: stack.pop() if i == '*' else stack.append(i)
#         return ''.join(stack)


# https://leetcode.com/problems/asteroid-collision/description/?envType=study-plan-v2&envId=leetcode-75
# 25 Asteroid Collision
# class Solution:
#     def asteroidCollision(self, asteroids: List[int]) -> List[int]:
#         j = 0
#         n = len(asteroids)
#
#         for i in range(n):
#             asteroid = asteroids[i]
#             while j > 0 and asteroids[j - 1] > 0 and asteroid < 0 and asteroids[j - 1] < abs(asteroid):
#                 j -= 1
#
#             if j == 0 or asteroid > 0 or asteroids[j - 1] < 0:
#                 asteroids[j] = asteroid
#                 j += 1
#             elif asteroids[j - 1] == abs(asteroid):
#                 j -= 1
#         return asteroids[:j]


# https://leetcode.com/problems/decode-string/description/?envType=study-plan-v2&envId=leetcode-75
# 26 Decode String
# class Solution:
#     def decodeString(self, s: str) -> str:
#         stack = []
#         curr_str = ""
#         curr_num = 0
#         for c in s:
#             if c.isdigit():
#                 curr_num = curr_num * 10 + int(c)
#             elif c == "[":
#                 stack.append(curr_num)
#                 stack.append(curr_str)
#                 # Reset the current number and current string
#                 curr_num = 0
#                 curr_str = ""
#             elif c == "]":
#                 prev_str = stack.pop()
#                 prev_num = stack.pop()
#                 curr_str = prev_str + curr_str * prev_num
#             else:
#                 curr_str += c
#         while stack:
#             curr_str = stack.pop() + curr_str
#         return curr_str


# https://leetcode.com/problems/number-of-recent-calls/description/?envType=study-plan-v2&envId=leetcode-75
# 27 Number of Recent Calls
# class RecentCounter:
#     def __init__(self):
#         self.counter =0
#         self.queue =deque()
#     def ping(self, t: int) -> int:
#         self.queue.append(t)
#         self.counter+=1
#         while self.queue[0] < t-3000:
#             self.queue.popleft()
#             self.counter-=1
#         return self.counter


# https://leetcode.com/problems/dota2-senate/solutions/?envType=study-plan-v2&envId=leetcode-75
# 28 Dota2 Senate
# class Solution:
#     def predictPartyVictory(self, senate: str) -> str:
#         num_r = 0
#         num_d = 0
#         # Count the number of D and Rs
#         for mem in senate:
#             if mem == "R":
#                 num_r += 1
#             else:
#                 num_d += 1
#         ban_r = 0
#         ban_d = 0
#         senate = list(senate)
#         floating_d_bans = 0
#         floating_r_bans = 0
#         # Loop until all D's are banned or all Rs are banned
#         while ban_d!=num_d and ban_r!=num_r:
#             for i,mem in enumerate(senate):
#                 # Member is R
#                 if mem == 'R':
#                     # There is a ban on R that isn't enforced yet
#                     if floating_r_bans > 0:
#                         floating_r_bans -= 1
#                         senate[i] = 'X'
#                         ban_r += 1
#                     else:
#                         floating_d_bans += 1
#                 if mem == 'D':
#                     if floating_d_bans > 0:
#                         floating_d_bans -= 1
#                         senate[i] = 'X'
#                         ban_d += 1
#                     else:
#                         floating_r_bans += 1
#         if ban_r == num_r:
#             return "Dire"
#         return "Radiant"


# https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/description/?envType=study-plan-v2&envId=leetcode-75
# 29 Delete the Middle Node of a Linked List
# class Solution(object):
#     def deleteMiddle(self, head):
#         """
#         :type head: Optional[ListNode]
#         :rtype: Optional[ListNode]
#         """
#         if head == None: return None
#         prev = ListNode(0)
#         prev.next = head
#         slow = prev
#         fast = head
#         while fast != None and fast.next != None:
#             slow = slow.next
#             fast = fast.next.next
#
#         slow.next = slow.next.next
#         return prev.next


# https://leetcode.com/problems/odd-even-linked-list/description/?envType=study-plan-v2&envId=leetcode-75
# 30 Odd Even Linked List
# class Solution(object):
#     def oddEvenList(self, head):
#         """
#         :type head: ListNode
#         :rtype: ListNode
#         """
#         if head == None or head.next == None : return head
#         odd = ListNode(0)
#         odd_ptr = odd
#         even = ListNode(0)
#         even_ptr = even
#         idx = 1
#         while head != None :
#             if idx % 2 == 0:
#                 even_ptr.next = head
#                 even_ptr = even_ptr.next
#             else:
#                 odd_ptr.next = head
#                 odd_ptr = odd_ptr.next
#             head = head.next
#             idx+=1
#         even_ptr.next = None
#         odd_ptr.next = even.next
#         return odd.next


# https://leetcode.com/problems/reverse-linked-list/description/?envType=study-plan-v2&envId=leetcode-75
# 31 Reverse Linked List
# class Solution:
#     def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         if  head==None or head.next==None:
#             return head
#         newHead = self.reverseList(head.next)
#         temp = head.next
#         temp.next = head
#         head.next = None
#         return newHead


# https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description/?envType=study-plan-v2&envId=leetcode-75
# 32 Maximum Twin Sum of a Linked List
# class Solution:
#     def pairSum(self, head: Optional[ListNode]) -> int:
#         l = []
#         curr = head
#         maxi = 0
#         while curr:
#             l.append(curr.val)
#             curr = curr.next
#         k = len(l)-1
#         left = 0
#         while left < k:
#             maxi = max(maxi,l[left]+l[k])
#             left += 1
#             k -= 1
#         return maxi


# https://leetcode.com/problems/maximum-depth-of-binary-tree/solutions/1769367/python3-recursive-dfs-explained/?envType=study-plan-v2&envId=leetcode-75
# 33 Maximum Depth of Binary Tree
# class Solution:
#     def maxDepth(self, root: Optional[TreeNode]) -> int:
#         def dfs(root, depth):
#             if not root: return depth
#             return max(dfs(root.left, depth + 1), dfs(root.right, depth + 1))
#
#         return dfs(root, 0)


# https://leetcode.com/problems/leaf-similar-trees/description/?envType=study-plan-v2&envId=leetcode-75
# 34 Leaf-Similar Trees
# class Solution:
#     def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
#         # Helper function to traverse the tree and collect leaf node values
#         def checkLeaf(root, ans):
#             # Base case: If the node is null, return
#             if not root:
#                 return
#
#             # Recursively check the left subtree
#             checkLeaf(root.left, ans)
#
#             # If the node is a leaf (both left and right children are null), add its value to the list
#             if not root.left and not root.right:
#                 ans.append(root.val)
#
#             # Recursively check the right subtree
#             checkLeaf(root.right, ans)
#
#         # Lists to store leaf values for each tree
#         ans1, ans2 = [], []
#
#         # Populate lists with leaf values using the helper function
#         checkLeaf(root1, ans1)
#         checkLeaf(root2, ans2)
#
#         # Check if the leaf sequences for both trees are equal
#         return ans1 == ans2


# https://leetcode.com/problems/count-good-nodes-in-binary-tree/description/?envType=study-plan-v2&envId=leetcode-75
# 35 Count Good Nodes in Binary Tree
# class Solution:
#     def goodNodes(self, root: TreeNode) -> int:
#         def helper(node, currMax):
#             if not node:
#                 return 0
#             if node.val >= currMax:
#                 cnt = 1
#             else:
#                 cnt = 0
#             currMax = max(currMax, node.val)
#             cnt += helper(node.left, currMax)
#             cnt += helper(node.right, currMax)
#             return cnt
#         return helper(root, root.val)


# https://leetcode.com/problems/path-sum-iii/description/?envType=study-plan-v2&envId=leetcode-75
# 36 Path Sum III
# class Solution:
#     def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
#         nums=defaultdict(int)
#         nums[0]=1
#         def dfs(root,tot):
#             cnt=0
#             if root:
#                 tot+=root.val
#                 cnt=nums[tot-targetSum]
#                 nums[tot]+=1
#                 cnt+=dfs(root.left,tot)+dfs(root.right,tot)
#                 nums[tot]-=1
#             return cnt
#         return dfs(root,0)


# https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/description/?envType=study-plan-v2&envId=leetcode-75
# 37 Longest ZigZag Path in a Binary Tree
# class Solution:
#     def longestZigZag(self, root: Optional[TreeNode]) -> int:
#         self.maxLength = 0
#         def solve(node, deep, dir):
#             self.maxLength = max(self.maxLength, deep)
#
#             if node.left is not None:
#                 solve(node.left, deep+1,'left') if dir != 'left' else solve(node.left, 1, 'left')
#             if node.right is not None:
#                 solve(node.right, deep+1, 'right') if dir != 'right' else solve(node.right, 1, 'right')
#         solve(root, 0, '')
#         return self.maxLength


# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=leetcode-75
# 38 Lowest Common Ancestor of a Binary Tree
# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         if root is None: return None
#         if root == p or root == q:
#             left = self.lowestCommonAncestor(root.left, p, q)
#             right = self.lowestCommonAncestor(root.right, p, q)
#             return root
#         left = self.lowestCommonAncestor(root.left, p, q)
#         right = self.lowestCommonAncestor(root.right, p, q)
#         if left is not None and right is not None: return root
#         if left is not None and left == root.left: return left
#         if right is not None and right == root.right: return right
#         if left is not None: return left
#         if right is not None: return right


# https://leetcode.com/problems/binary-tree-right-side-view/description/?envType=study-plan-v2&envId=leetcode-75
# 39 Binary Tree Right Side View
# from collections import deque
# class Solution:
#     def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
#         queue = deque()
#         if root is None:
#             return []
#         if root.left is None and root.right is None:
#             return [root.val]
#         result = []
#         queue.append(root)
#         while queue:
#             child_queue = deque()
#             prev = -1
#             while queue:
#                 curr = queue.popleft()
#                 if curr.left is not None:
#                     child_queue.append(curr.left)
#                 if curr.right is not None:
#                     child_queue.append(curr.right)
#                 prev = curr
#             result.append(prev.val)
#             queue = child_queue
#         return result


# https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description/?envType=study-plan-v2&envId=leetcode-75
# 40 Maximum Level Sum of a Binary Tree
# class Solution:
#     def maxLevelSum(self, root: Optional[TreeNode]) -> List[float]:
#         level_sum = defaultdict(int)
#
#         for lvl, val in self.inorder(root):
#             level_sum[lvl + 1] += val
#
#         return max(level_sum, key=lambda x: (level_sum[x], -x))
#
#     @classmethod
#     def inorder(cls, tree: TreeNode | None, level: int = 0):
#         if tree is not None:
#             yield from cls.inorder(tree.left, level + 1)
#             yield level, tree.val
#             yield from cls.inorder(tree.right, level + 1)


# https://leetcode.com/problems/search-in-a-binary-search-tree/description/?envType=study-plan-v2&envId=leetcode-75
# 41 Search in a Binary Search Tree
# class Solution:
#     def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
#         # null check && considering serching value is not available in the tree
#         if not root:
#             return
#         # considering the serching value is in right side of the tree
#         if val>root.val:
#             return self.searchBST(root.right,val)
#         # considering the serching value is in right side of the tree
#         elif val<root.val:
#             return self.searchBST(root.left,val)
#         #return if value exist
#         return root


# https://leetcode.com/problems/delete-node-in-a-bst/description/?envType=study-plan-v2&envId=leetcode-75
# 42 Delete Node in a BST
# class Solution:
#     def smallestDescendant(self, root):
#         while root.left:
#             root = root.left
#         return root
#     def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
#         if not root:
#             return
#         else:
#             # Checks for Lesser value
#             if root.val > key:
#                 root.left = self.deleteNode(root.left, key)
#             # Checks for Greater value
#             elif root.val < key:
#                 root.right = self.deleteNode(root.right, key)
#             # Checks if we have reached the node we want to delete
#             else:
#                 # First we check if only 1 child node is present
#                 if not root.left:
#                     root = root.right
#                 elif not root.right:
#                     root = root.left
#                 else:
#                 # If both the children are present then we find the
#                 # smallest child in the right child and assign it's
#                 # value to node and recursively delete that child
#                 # until we have reach node with 1 child or leaf node
#                     temp = self.smallestDescendant(root.right)
#                     root.val = temp.val
#                     root.right = self.deleteNode(root.right, temp.val)
#             return root


# https://leetcode.com/problems/keys-and-rooms/?envType=study-plan-v2&envId=leetcode-75
# 43 Keys and Rooms
# class Solution:
#     def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
#         k = new = set(rooms[0])
#         while len(k) < len(rooms) - 1:
#             new = reduce(
#                 lambda a, b: a.union(b), (set(rooms[n]) for n in new), set()
#             ) - k.union({0})
#             if not new:
#                 return False
#             k.update(new)
#         return True


# https://leetcode.com/problems/number-of-provinces/description/?envType=study-plan-v2&envId=leetcode-75
# 44 Number of Provinces
# class UnionFind(object):
#
#     def __init__(self, parents: list, rank: list):
#         self.parents = parents
#         self.rank = rank
#     def find(self, node):
#         parent = node
#         while parent != self.parents[parent]:
#             # don't get confused by that line
#             # this is just a path compression
#             self.parents[parent] = self.parents[self.parents[parent]]
#             parent = self.parents[parent]
#         return parent
#     def union(self, node1, node2):
#         first, second = self.find(node1), self.find(node2)
#         if first == second: return 0
#         if self.rank[first] > self.rank[second]:
#             self.parents[second] = first
#             self.rank[first] += self.rank[second]
#         else:
#             self.parents[first] = second
#             self.rank[second] += self.rank[first]
#         return 1
# class Solution:
#     def findCircleNum(self, isConnected: List[List[int]]) -> int:
#         parents = [node for node in range(len(isConnected))]
#         rank = [1] * len(parents)
#         unionFind = UnionFind(parents=parents, rank=rank)
#         result = len(parents)
#         for edge1 in range(len(isConnected)):
#             for otherEdge in range(len(isConnected)):
#                 if isConnected[edge1][otherEdge] == 1:
#                     result -= unionFind.union(edge1, otherEdge)
#         return result


# https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/description/?envType=study-plan-v2&envId=leetcode-75
# 45 Reorder Routes to Make All Paths Lead to the City Zero
# class Solution:
#     def minReorder(self, n: int, connections: List[List[int]]) -> int:
#         reachables = {0}
#         reorder = 0
#         stack = []
#         while connections:
#             a, b = connections.pop()
#             if b in reachables:
#                 reachables.add(a)
#             elif a in reachables:
#                 reachables.add(b)
#                 reorder += 1
#             else:
#                 stack.append([a,b])
#             if len(connections) == 0:
#                 connections = stack
#                 stack = []
#         return reorder


# https://leetcode.com/problems/evaluate-division/description/?envType=study-plan-v2&envId=leetcode-75
# 46 Evaluate Division
# class Solution:
#     def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
#         #   No division by zero
#         adjacencyList = collections.defaultdict(
#             list)  # A defaultdict is a dictionary-like object that automatically initializes values for nonexistent keys
#         for i, eq in enumerate(equations):
#             a, b = eq  # Each equation has two values
#             adjacencyList[a].append([b, values[i]])  # Append [b,value(a/b)]
#             adjacencyList[b].append([a, 1 / values[i]])  # b/a will be equal to 1 / (a/b)
#         print(adjacencyList)
#         def bfs(src, trg):
#             if src not in adjacencyList or trg not in adjacencyList:
#                 return -1
#             q = deque()
#             visited = set()
#             q.append([src, 1])  # I'll append a node with the weight upto that node
#             visited.add(src)
#             while q:
#                 n, w = q.popleft()  # Neighbor, Weight
#                 if n == trg:
#                     return w
#                 for neighbor, weight in adjacencyList[n]:  # Iterating over the adjacency List of that particular node
#                     if neighbor not in visited:
#                         q.append([neighbor, w * weight])
#                         visited.add(n)
#             return -1
#         return [bfs(query[0], query[1]) for query in queries]


# https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/description/?envType=study-plan-v2&envId=leetcode-75
# 47 Nearest Exit from Entrance in Maze
# class Solution:
#     def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
#         m = len(maze)
#         n = len(maze[0])
#         queue = collections.deque()
#         queue.append((entrance[0], entrance[1], 0))
#         visited = set()
#         visited.add((entrance[0], entrance[1]))
#         while queue:
#             for _ in range(len(queue)):
#                 r, c, level = queue.popleft()
#                 if [r, c] != entrance:
#                     if r == 0 or r == m - 1 or c == 0 or c == n - 1:
#                         return level
#                 for nr, nc in [(r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)]:
#                     if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited and maze[nr][nc] == '.':
#                         queue.append((nr, nc, level + 1))
#                         visited.add((nr, nc))
#         return -1


# https://leetcode.com/problems/rotting-oranges/description/?envType=study-plan-v2&envId=leetcode-75
# 48 Rotting Oranges
# class Solution:
#     def orangesRotting(self, grid: List[List[int]]) -> int:
#         m, n = len(grid), len(grid[0])
#         visited = grid
#         q = collections.deque()
#         countFreshOrange = 0
#         for i in range(m):
#             for j in range(n):
#                 if visited[i][j] == 2:
#                     q.append((i, j))
#                 if visited[i][j] == 1:
#                     countFreshOrange += 1
#         if countFreshOrange == 0:
#             return 0
#         if not q:
#             return -1
#         minutes = -1
#         dirs = [(1, 0), (-1, 0), (0, -1), (0, 1)]
#         while q:
#             size = len(q)
#             while size > 0:
#                 x, y = q.popleft()
#                 size -= 1
#                 for dx, dy in dirs:
#                     i, j = x + dx, y + dy
#                     if 0 <= i < m and 0 <= j < n and visited[i][j] == 1:
#                         visited[i][j] = 2
#                         countFreshOrange -= 1
#                         q.append((i, j))
#             minutes += 1
#
#         if countFreshOrange == 0:
#             return minutes
#         return -1


# https://leetcode.com/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=leetcode-75
# 49 Kth Largest Element in an Array
# class Solution:
#     def findKthLargest(self, nums, k):
#         return sorted(nums, reverse=True)[k-1]


# https://leetcode.com/problems/smallest-number-in-infinite-set/description/?envType=study-plan-v2&envId=leetcode-75
# 50 Smallest Number in Infinite Set
# class SmallestInfiniteSet:
#
#     def __init__(self):
#         self.min = 1
#         self.added = set()
#
#     def popSmallest(self) -> int:
#         to_pop = self.min
#         if len(self.added) > 0:
#             to_pop = min(min(self.added), to_pop)
#
#         self.added.discard(to_pop)
#         if to_pop == self.min:
#             self.min += 1
#
#         return to_pop
#
#     def addBack(self, num: int) -> None:
#         if num < self.min:
#             self.added.add(num)


# https://leetcode.com/problems/maximum-subsequence-score/description/?envType=study-plan-v2&envId=leetcode-75
# 51 Maximum Subsequence Score
# import heapq
# class Solution:
#     def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
#         lst=list(zip(nums2,nums1))
#         lst.sort(key=lambda x:(-x[0],-x[1]))
#         flst=[]
#         heapq.heapify(flst)
#         i=0
#         sm=0
#         ef=float("infinity")
#         prd=float("-infinity")
#         while i<k:
#             x=lst.pop(0)
#             heapq.heappush(flst,x[1])
#             ef=min(ef,x[0])
#             sm+=x[1]
#             i+=1
#         prd=max(prd,sm*ef)
#         while lst:
#             x=heapq.heappop(flst)
#             sm-=x
#             y=lst.pop(0)
#             heapq.heappush(flst,y[1])
#             ef=min(ef,y[0])
#             sm+=y[1]
#             prd=max(prd,sm*ef)
#         return prd


# https://leetcode.com/problems/total-cost-to-hire-k-workers/description/?envType=study-plan-v2&envId=leetcode-75
# 52 Total Cost to Hire K Workers
# class Solution:
#     def totalCost(self, costs: list[int], k: int, candidates: int) -> int:
#         n = len(costs)
#         l, r = candidates, n - candidates - 1
#         l_bucket = zip(range(l), repeat(0))
#         r_bucket = zip(range(max(l, r + 1), n), repeat(1))
#         hq = [(costs[idx], bucket) for idx, bucket in chain(l_bucket, r_bucket)]
#         heapq.heapify(hq)
#         cost = 0
#         for _ in range(k):
#             c, b = heapq.heappop(hq)
#             cost += c
#             if l > r: continue
#             idx = (l, r)[b]
#             l, r = (l, r - 1) if b else (l + 1, r)
#             heapq.heappush(hq, (costs[idx], b))
#         return cost


# https://leetcode.com/problems/guess-number-higher-or-lower/solutions/?envType=study-plan-v2&envId=leetcode-75
# 53 Guess Number Higher or Lower
# class Solution:
#     def guessNumber(self, n: int) -> int:
#         low, high = 1, n
#         while low <= high:
#             mid = (low + high) // 2
#             G = guess(mid)
#             if G == 0:
#                 return mid
#             elif G == -1:
#                 high = mid - 1
#             else:
#                 low = mid + 1
#         return mid


# https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description/?envType=study-plan-v2&envId=leetcode-75
# 54 Successful Pairs of Spells and Potions
# class Solution:
#     def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
#         n, m = len(spells), len(potions)
#         pairs = [0]*n
#         potions.sort()
#         for i in range(n):
#             spell = spells[i]
#             left = 0
#             right = m - 1
#             while left <= right:
#                 mid = left+(right-left)//2
#                 product = spell*potions[mid]
#                 if product >= success:
#                     right = mid - 1
#                 else:
#                     left = mid + 1
#             pairs[i] = m - left
#         return pairs


# https://leetcode.com/problems/find-peak-element/description/?envType=study-plan-v2&envId=leetcode-75
# 55 Find Peak Element
# class Solution:
#     def findPeakElement(self, nums: List[int]) -> int:
#         n = len(nums)
#         low, high = 0, n - 1
#         while low <= high:
#             mid = ((high - low) // 2) + low
#             if mid < n - 1 and nums[mid] < nums[mid + 1]:
#                 low = mid + 1
#             elif mid > 0 and nums[mid] < nums[mid - 1]:
#                 high = mid - 1
#             else:
#                 return mid


# https://leetcode.com/problems/koko-eating-bananas/description/?envType=study-plan-v2&envId=leetcode-75
# 56 Koko Eating Bananas
# class Solution:
#     def f(self,p,k):
#         t = 0
#         for i in p:
#             t+=ceil(i/k)
#         return t
#     def minEatingSpeed(self, piles: List[int], h: int) -> int:
#         s = 1
#         e = max(piles)
#
#         while(s<=e):
#             m = s + (e-s)//2
#             if self.f(piles,m)>h:
#                 s = m+1
#             else:
#                 e = m-1
#         return s


# https://leetcode.com/problems/letter-combinations-of-a-phone-number/solutions/?envType=study-plan-v2&envId=leetcode-75
# 57 Letter Combinations of a Phone Number
# class Solution:
#     def letterCombinations(self, digits: str) -> List[str]:
#         if not digits:
#             return []
#         m = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
#         res = []
#
#         def backtrack(curr,i):
#             if len(curr)==len(digits):
#                 res.append(curr)
#                 return
#             for ch in m[digits[i]]:
#                 backtrack(curr+ch,i+1)
#
#         backtrack("",0)
#         return res


# https://leetcode.com/problems/combination-sum-iii/description/?envType=study-plan-v2&envId=leetcode-75
# 58 Combination Sum III
# class Solution:
#     def combinationSum3(self, k: int, n: int) -> List[List[int]]:
#         return [
#             x
#             for x in [
#                 reduce(lambda x, y: x + y, s)
#                 for s in product(*[[[], [i]] for i in range(1, 10)])
#             ]
#             if len(x) == k and sum(x) == n
#         ]


# https://leetcode.com/problems/n-th-tribonacci-number/description/?envType=study-plan-v2&envId=leetcode-75
# 59 N-th Tribonacci Number
# class Solution:
#     def tribonacci(self, n: int) -> int:
#         if n < 3:
#             return 1 if n else 0
#         a, b, c = 0, 1, 1
#         for _ in range(n - 2):
#             a, b, c = b, c, a + b + c
#         return c


# https://leetcode.com/problems/min-cost-climbing-stairs/description/?envType=study-plan-v2&envId=leetcode-75
# 60 Min Cost Climbing Stairs
# class Solution:
#     def minCostClimbingStairs(self, cost: List[int]) -> int:
#         n = len(cost)
#         for i in range(2, len(cost)):
#             cost[i] += min(cost[i - 1], cost[i - 2])
#         return min(cost[n - 1], cost[n - 2])


# https://leetcode.com/problems/house-robber/description/?envType=study-plan-v2&envId=leetcode-75
# 61 House Robber
# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         return reduce(lambda a, x: (a[1], max(a[1], a[0] + x)), nums, (0, 0))[1]


# https://leetcode.com/problems/domino-and-tromino-tiling/description/?envType=study-plan-v2&envId=leetcode-75
# 62 Domino and Tromino Tiling
# class Solution(object):
#     def numTilings(self, n):
#         dp, dpa = [1, 2] + [0] * n, [1] * n
#         for i in range(2, n):
#             dp[i] = (dp[i - 1] + dp[i - 2] + dpa[i - 1] * 2) % 1000000007
#             dpa[i] = (dp[i - 2] + dpa[i - 1]) % 1000000007
#         return dp[n - 1]


# https://leetcode.com/problems/unique-paths/description/?envType=study-plan-v2&envId=leetcode-75
# 63 Unique Paths
# class Solution:
#     def uniquePaths(self, m: int, n: int) -> int:
#         oldrow = [1] * n
#         for i in range(m - 1):
#             newrow = [1] * n
#             for j in range(n - 2, -1, -1):
#                 newrow[j] = newrow[j + 1] + oldrow[j]
#             oldrow = newrow
#         return oldrow[0]


# https://leetcode.com/problems/longest-common-subsequence/description/?envType=study-plan-v2&envId=leetcode-75
# 64 Longest Common Subsequence
# class Solution:
#     def longestCommonSubsequence(self, text1: str, text2: str) -> int:
#         #LCS is a standard question for DP
#         n=len(text1)
#         m=len(text2)
#         dp=[[-1]*m for _ in range(n)]
#         def lcs(i, j):
#             nonlocal n, m, dp
#             if i>=n or j>=m: return 0
#             if dp[i][j]!=-1: return dp[i][j]
#             if text1[i]==text2[j]:
#                 dp[i][j]=1+lcs(i+1, j+1)
#                 return dp[i][j]
#             else:
#                 dp[i][j]= max(lcs(i, j+1), lcs(i+1, j))
#                 return dp[i][j]
#         return lcs(0, 0)


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/?envType=study-plan-v2&envId=leetcode-75
# 65 Best Time to Buy and Sell Stock with Transaction Fee
# class Solution:
#     def maxProfit(self, prices: List[int], fee: int) -> int:
#         buy = float('-inf')
#         sell = 0
#         for price in prices:
#             buy = max(buy, sell - price)
#             sell = max(sell, buy + price - fee)
#         return sell


# https://leetcode.com/problems/edit-distance/description/?envType=study-plan-v2&envId=leetcode-75
# 66 Edit Distance
# class Solution:
#   def minDistance(self, word1: str, word2: str) -> int:
#     m = len(word1)
#     n = len(word2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#
#     for i in range(1, m + 1):
#       dp[i][0] = i
#
#     for j in range(1, n + 1):
#       dp[0][j] = j
#
#     for i in range(1, m + 1):
#       for j in range(1, n + 1):
#         if word1[i - 1] == word2[j - 1]:
#           dp[i][j] = dp[i - 1][j - 1]
#         else:
#           dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
#
#     return dp[m][n]


# https://leetcode.com/problems/counting-bits/?envType=study-plan-v2&envId=leetcode-75
# 67 Counting Bits
# class Solution:
#     def countBits(self, n: int) -> List[int]:
#         ans = [0] * (n + 1)
#         for i in range(1, n + 1):
#             ans[i] = ans[i >> 1] + (i & 1)
#         return ans


# https://leetcode.com/problems/single-number/description/?envType=study-plan-v2&envId=leetcode-75
# 68 Single Number
# class Solution:
#     def singleNumber(self, nums: List[int]) -> int:
#         xor = 0
#         for num in nums:
#             xor ^= num
#
#         return xor


# https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/description/?envType=study-plan-v2&envId=leetcode-75
# 69 Minimum Flips to Make a OR b Equal to c
# class Solution:
#     def minFlips(self, a: int, b: int, c: int) -> int:
#         m = (a|b)^c
#         x = a&m
#         y = b&m
#         z = (~(a|b))&m
#         return sum(v.bit_count() for v in (x,y,z))


# https://leetcode.com/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=leetcode-75
# 70 Implement Trie (Prefix Tree)
# class Trie:
#
#     def __init__(self):
#         self.t = []
#
#     def insert(self, word: str) -> None:
#         bisect.insort(self.t, word)
#
#     def search(self, word: str) -> bool:
#         i = bisect.bisect(self.t, word)
#         return i>0 and self.t[i-1] == word
#
#     def startsWith(self, prefix: str) -> bool:
#         j = bisect.bisect_left(self.t, prefix)
#         return j<len(self.t) and self.t[j].startswith(prefix)


# https://leetcode.com/problems/search-suggestions-system/description/?envType=study-plan-v2&envId=leetcode-75
# 71 Search Suggestions System
# class Solution:
#     def suggestedProducts(self, p: List[str], s: str) -> List[List[str]]:
#         p.sort()
#         ret = []
#         for i in range(1, len(s) + 1):
#             j=bisect_left(p, s[:i])
#             ret.append([x for x in p[j:j+3] if x.startswith(s[:i])])
#         return ret


# https://leetcode.com/problems/non-overlapping-intervals/description/?envType=study-plan-v2&envId=leetcode-75
# 72
# class Solution:
#     def eraseOverlapIntervals(self, points: List[List[int]]) -> int:
#         ans = 0
#         arrow = -math.inf
#         for point in sorted(points, key = lambda x:x[1]):
#             if(point[0] >= arrow):
#                 arrow = point[1]
#             else:
#                 ans+=1
#         return ans


# https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/?envType=study-plan-v2&envId=leetcode-75
# 73 Minimum Number of Arrows to Burst Balloons
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




# https://leetcode.com/problems/online-stock-span/description/?envType=study-plan-v2&envId=leetcode-75
# 74 Daily Temperatures
# class StockSpanner:
#     def __init__(self):
#         self.memo = []
#         self.timestamp = 0
#
#     def next(self, price: int) -> int:
#         self.timestamp += 1
#         record = {'ts': self.timestamp, 'price': price}
#         while (self.memo and price >= self.memo[-1]['price']):
#             self.memo.pop()
#         self.memo.append(record)
#         if len(self.memo) > 1:
#             return (self.memo[-1]['ts'] - self.memo[-2]['ts'])
#         else:
#             return self.timestamp


# https://leetcode.com/problems/daily-temperatures/description/?envType=study-plan-v2&envId=leetcode-75
# 75 Online Stock Span
# class Solution:
#     def dailyTemperatures(self, t: List[int]) -> List[int]:
#         n = len(t)
#         if n == 1: return [0]
#         ans = [0] * n
#         st = [n - 1]
#         for i in range(n - 2, -1, -1):
#             while st and t[i] >= t[st[-1]]:
#                 st.pop()
#             if st:
#                 ans[i] = st[-1] - i
#             st.append(i)
#         return ans
