# Heapsort — lecture 6 notes

Sorts an array of n numbers in place-ish. Idea: use a priority queue.

1. Insert all n elements into the priority queue (n inserts).
2. Repeatedly remove the minimum and write it to the output (n extract-min operations).

The queue never holds more than n elements. Copying the input and writing the output costs O(n) in total.

TODO: ask TA whether in-place version changes anything
