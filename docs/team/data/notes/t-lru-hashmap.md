Hash map (dictionary) -- what I need to remember

Holds n entries. Worst case ignored, everything below is the *average* (expected) cost with a decent hash function.

- lookup by key: O(1) on average
- put(key, value): O(1) on average (overwrites if the key is already there)
- delete(key): O(1) on average
- iterate over all entries: O(n), order not guaranteed

the load factor and resizing are in the other notes. Java HashMap, python dict, C++ unordered_map, same idea.
