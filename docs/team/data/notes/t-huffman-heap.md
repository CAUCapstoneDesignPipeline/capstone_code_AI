# Binary heap notes

A binary heap keeps n items in an array that is read as a complete binary tree. In a min-heap every parent <= its children. (max-heap = same thing with the comparison flipped)

What you can do with it:
- push(x): put x at the end of the array, then sift it up. Costs O(log n), n = number of items currently in the heap.
- pop the smallest: take the root, move the last item to the root, sift down. Costs O(log n).
- look at the smallest without removing it: O(1)
- heapify a whole array that is already there: O(n)

(The prof calls "pop the smallest" extract-min, I write it my way so I remember what it does.)

Index stuff, 0-indexed: parent of i is (i-1)//2, children are 2i+1 and 2i+2.
Python's heapq is a min-heap, java PriorityQueue also.
