# 0/1 knapsack (DP)

dp[i][w] = best value using the first i items with capacity w.
dp[i][w] = max(dp[i-1][w], dp[i-1][w - wt_i] + val_i)
Table has n * W cells, each filled in O(1), so O(nW). This is pseudo-polynomial because W is a number, not the input length.
Can be shrunk to one row if we loop w downwards.

- greedy by value/weight ratio works only for the fractional version
- homework 5 q3 is a variation with two capacities
