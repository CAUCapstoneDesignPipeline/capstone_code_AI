# Hash table cheat sheet

Chained hashing, resize when it gets too full. Costs are for a table that holds n entries. I only wrote down the *average* case.

| operation | what it does | average cost |
|---|---|---|
| put(key, value) | insert, or overwrite if key exists | O(1) |
| probe(key) | find the value for key, or say nothing is there | ??? (look this up again, forgot) |
| remove(key) | delete the entry | O(1) |

- table doubles when load factor > 0.75
- python dict is a hash table too (open addressing, NOT chaining)
- good hash function matters!!
