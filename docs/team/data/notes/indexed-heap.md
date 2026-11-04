# 인덱스 있는 이진 최소 힙 (Indexed Binary Min Heap)

자료구조 과제 3 하면서 정리.

보통 이진 힙은 배열로 만든 완전 이진 트리고, 부모 키 ≤ 자식 키.
그냥 이진 힙은 "정점 v 가 지금 힙의 어디 있지?"를 모름 → 키를 바꾸려면 O(n) 탐색.

**인덱스 있는 버전**은 pos[v] 배열로 각 원소의 위치를 같이 들고 다닌다.
- 원소끼리 키를 비교할 수 있어야 한다.
- 그래서 우선순위 큐로 쓸 수 있다.

## 연산 (힙 크기 n, 최악)
| 연산 | 비용 |
|---|---|
| insert | O(log n) |
| extract-min | O(log n) |
| decrease-key | O(log n) |

insert 는 맨 뒤에 넣고 위로 올리기(sift-up), extract-min 은 루트를 빼고 마지막 원소를 올린 뒤 아래로(sift-down).
decrease-key 는 pos[v] 로 바로 찾아서 sift-up 만 하면 된다.
