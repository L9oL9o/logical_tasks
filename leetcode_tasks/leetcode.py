# class Solution:
#     def romanToInt(self, s: str) -> int:

class Solution:
    def intToRoman(self, num: int) -> str:

        roman_values = {
            1: "I",
            5: "V",
            10: "X",
            50: "L",
            100: "C",
            500: "D",
            1000: "M",
        }
        result = 0
        prev_value = 0
        for char in s:
            value = roman_values[char]
            if value > prev_value:
                # If a smaller value precedes a larger value, subtract the smaller value
                result += value - 2 * prev_value
            else:
                result += value
            prev_value = value
        return result
