# Huffman coding (week 9)

Source: lecture 14 + textbook ch. on greedy algorithms. Typed this on the bus so sorry for typos.

- Goal: build an optimal prefix code for an alphabet of k distinct symbols, each symbol comes with a frequency.
- We keep a min-priority queue of tree nodes, keyed by frequency. Only two things are ever done to it: push a node, and pop the smallest node (the one with lowest frequency).
- Step 1: make one leaf per symbol and push all k leaves into the queue. That is k pushes.
- Step 2: repeat k - 1 times: pop the smallest, pop the smallest again, make a parent node whose frequency is the sum of the two, push the parent. So step 2 does 2(k-1) pops and k-1 pushes.
- The queue never holds more than k nodes at once (each merge removes two nodes and adds one).
- Step 3: walk the finished tree one time to read off the codewords. This walk is O(k) and nothing else happens after it.
- the example from the slides: a:45 b:13 c:12 d:16 e:9 f:5  -> TODO redo it by hand before the midterm
- prof said arithmetic coding is NOT on the exam
- ask TA about ties in frequency (does it matter?)
