# B+ tree index

- balanced tree, all records live in the leaves, leaves are chained in a linked list
- For an index with $n$ keys:
  - point lookup by key: $O(\log n)$ (height is really $\log_B n$ but B = fan-out is a constant for us)
  - range scan from key a to key b: $O(\log n + r)$, r = number of results
  - insert: $O(\log n)$
  - delete: $O(\log n)$
- After the first descent a range scan is just following the next pointers in the leaf list.
- Space: $\sum_{i} n / B^i$ is a geometric series so the internal nodes are only a small fraction, total $O(n)$.
- fan-out in real DBs is a few hundred (8KB pages)
