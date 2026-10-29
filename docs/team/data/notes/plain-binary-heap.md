# 이진 힙 (Binary Heap)

완전 이진 트리 + 힙 속성. 배열 하나로 만든다.

우선순위 큐 구현에 제일 많이 쓴다.

- insert: O(log n)
- extract-min: O(log n)
- 원소가 배열의 어디 있는지 따로 기록하지 않아서 이진 힙은 decrease-key 를 지원하지 않는다.
