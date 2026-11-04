# Index nested-loop join (DB ch. 12)

Setup: join outer table $R$ with inner table $S$ on `R.sid = S.id`. $|R| = N$ rows, $|S| = M$ rows. `S.id` is the primary key so there is an index on it.

Algorithm:
- scan $R$ sequentially, one row at a time. The scan alone costs $O(N)$.
- for each outer row do one *point lookup by key* in the index on `S.id` to find the matching inner row. One lookup per outer row => $N$ lookups in total.
- since `S.id` is a PK, an outer row matches at most one inner row, so writing out the result is another $O(N)$ in total.
- the index has one entry per inner row, so it never holds more than $M$ keys.

misc:
- project 2 is due friday!!
- the textbook uses $\bowtie$ for join, I keep forgetting
- buffer pool size not considered here (ask TA?)
