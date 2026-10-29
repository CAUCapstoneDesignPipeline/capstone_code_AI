# Topological sort with DFS (iterative)

there are N tasks and D dependency edges (a -> b means a before b).

1. keep a colour array (white / gray / black), nothing fancy
2. explicit stack for the DFS: push a task when we first reach it, pop it when we backtrack
3. each task is pushed once and popped once, so N pushes and N pops
4. a task gets appended to the output list when it is popped (finish order), then reverse the list at the end
5. scanning the adjacency lists of all tasks costs O(N + D) in total, the only other work

Cycle => gray node seen again => no valid order.
Reminder: Kahn's algorithm is the BFS-ish alternative, uses in-degree counts.
also buy milk lol
