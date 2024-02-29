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















































# https://leetcode.com/problems/online-stock-span/description/?envType=study-plan-v2&envId=leetcode-75
# 74
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
# 75
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
