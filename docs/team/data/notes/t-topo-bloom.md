# Bloom filter

Space-efficient set membership test for a set of n items, m bits, k hash functions.

- add(x): set k bits. O(k)
- might_contain(x): check k bits. False positives possible, false negatives never.
- No deletion (use a counting bloom filter if you need it)
- false positive rate ~ (1 - e^{-kn/m})^k
- used in databases / web caches to skip useless disk lookups
