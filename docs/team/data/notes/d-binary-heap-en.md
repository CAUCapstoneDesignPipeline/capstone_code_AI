# Binary heap

A complete binary tree stored in an array; parent <= children.

| operation | cost (heap holds n items) |
|---|---|
| insert | O(log n) |
| extract-min | O(log n) |
| find-min | O(1) |

Can be used as a priority queue. Insert = append + sift-up. Extract-min = swap root with last + sift-down.
