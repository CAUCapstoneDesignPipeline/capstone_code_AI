hash join -- quick notes (DB lecture 7)

two tables: build side has m rows, probe side has p rows. (build = the smaller one, in theory)

phase 1 (build): go through the build table once and put every row into an in-memory hash table keyed on the join column. m puts.
phase 2 (probe): go through the other table, and for each row probe the hash table with its join key. p probes.
the hash table only ever gets build-side rows so it has at most m entries.
Reading both tables sequentially costs O(m + p), that is the only other work.

I am only thinking about the average case here because the prof said the worst case (all keys collide) is silly.

random: the lab for grace hash join (partitioning to disk) is next week, not today's topic
