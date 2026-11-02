LRU cache (leetcode 146 + OS lecture)

- capacity = c entries. Input is a sequence of q requests, each request names one key.
- impl: a hash map from key -> list node, plus a doubly linked list ordered by recency (front = most recently used)
- for every single request:
  1. lookup by key in the map. 1 lookup per request => q lookups total
  2. then put the key into the map. We do this every time (hit = refresh, miss = new entry). q puts total
  3. if the cache is over capacity, evict the tail: delete the tail's key from the map. This can happen at most once per request => at most q deletes total
  4. move/insert the node at the front of the linked list: O(1) pointer changes per request, so O(q) over the whole run, this is the only work outside the map
- the map never holds more than c entries (we evict as soon as it would be c+1)
- FIFO is simpler but evicts hot pages, see thrashing example in slides
- typo in the leetcode starter code, `capcity`
