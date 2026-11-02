# Cache summary (for the exam)

- LRU = evict the least recently used entry. FIFO = evict oldest inserted.
- LRU cache with capacity c over q requests: the hash map from my hashmap note can be used as the key index of the LRU cache, because it has lookup by key, put and delete, which is everything the cache needs from it.
- With that, one whole run costs O(q) on average (q lookups + q puts + at most q deletes + O(q) list moves).
- Belady's anomaly only happens for FIFO (not for LRU) -> exam question last year
