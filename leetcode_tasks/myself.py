# 7 1_JANUARY 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/length-of-last-word |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def lengthOfLastWord(self, s: str) -> int:
#         words = s.split()
#         if words:
#             s = words[-1]
#             return len(s)

# ~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def lengthOfLastWord(self, s: str) -> int:
#         return len(s.strip().split(' ')[-1])



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# # https://leetcode.com/problems/plus-one/ |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~ MYSELF ~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def plusOne(self, digits: List[int]) -> List[int]:
#         if digits:
#             if digits[-1] <= 8:
#                 digits[-1] += 1
#                 return digits
#             elif digits[-1] == 9:
#                 digits[-1] = 1
#                 digits.append(0)
#                 return digits

# ~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def plusOne(self, digits: List[int]) -> List[int]:
#         n = len(digits)
#
#         # Start from the least significant digit
#         for i in range(n - 1, -1, -1):
#             # Increment the current digit
#             digits[i] += 1
#
#             # If there is no carry, we are done
#             if digits[i] < 10:
#                 break
#             else:
#                 # Set the current digit to 0 and continue to the next digit
#                 digits[i] = 0
#
#         # If there is a carry after processing all digits, add a new leading digit
#         if digits[0] == 0:
#             digits.insert(0, 1)
#
#         return digits

# ~~~~~~~~~~~~~~~~~ LEETCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def plusOne(self, digits: List[int]) -> List[int]:
#         digits_str = [str(i) for i in digits]
#         num = str(int("".join(digits_str)) + 1)
#         return [int(i) for i in num]
