# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/?envType=daily-question&envId=2024-01-28 |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# class Solution:
#     def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
#         rows, cols = len(matrix), len(matrix[0])
#
#         # Calculate the prefix sum for each row
#         for row in matrix:
#             for j in range(1, cols):
#                 row[j] += row[j - 1]
#
#         count = 0  # Variable to store the count of submatrices with the target sum
#
#         # Iterate over all possible pairs of columns (left and right)
#         for left in range(cols):
#             for right in range(left, cols):
#                 prefix_sum_count = {0: 1}  # Hash table to store prefix sum count
#                 current_sum = 0  # Variable to store the current sum
#
#                 # Iterate over all rows and calculate the current sum
#                 for i in range(rows):
#                     # Calculate the sum of elements in the submatrix formed by rows 0 to i and columns left to right
#                     current_sum += matrix[i][right] - (matrix[i][left - 1] if left > 0 else 0)
#
#                     # If there is a prefix sum that makes the current sum - target, increment the count
#                     count += prefix_sum_count.get(current_sum - target, 0)
#
#                     # Increment the prefix sum count for the current sum
#                     prefix_sum_count[current_sum] = prefix_sum_count.get(current_sum, 0) + 1
#
#         return count