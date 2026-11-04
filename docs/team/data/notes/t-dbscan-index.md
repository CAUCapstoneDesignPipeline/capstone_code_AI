Here is what I know about the k-d tree from the library we use in the lab (the one from the course repo, not scikit-learn). Building it from n points takes O(n log n). It answers a nearest-neighbour query, and the k-nearest variant, in O(log n) on average when the dimension is low, and it gets much worse once the dimension is high, which is the curse of dimensionality again.

The thing the TA stressed during the lab is that this class does NOT implement radius or range queries. There is simply no method that returns all the points within some distance of a query point; the only queries are nearest and k-nearest. So whatever we build on it can only ask for a fixed number of closest points, never for everything inside a ball. He said we would have to write that ourselves, which is not part of the assignment.

Unrelated, but the lab report is due on the 14th and has to be in pdf.
