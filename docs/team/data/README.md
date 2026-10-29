# 조건부 결론 대표 사례 (안건 4)

R1(구현 적용)·R2(비용 합성)의 대표 입력·기대 결과·반례. 팀이 같은 노트를 보고 같은 전제와 결론을 말할 수 있게 고정해 둔 자료다.

```
notes/<이름>.md      노트 원문 (학생 노트처럼 쓴 설명용 예시)
facts/<이름>.json    그 노트에서 사람이 적은 계산용 사실. quote 는 원문 그대로 (위치는 코드가 찾는다)
cases.json           사례: 쓰는 노트와 기대 결과. aliases 는 LLM 이 같은 대상을 달리 부를 수 있는 key
```

| 사례 | 장면 | 기대 |
| --- | --- | --- |
| `dijkstra-indexed-heap` | 정상 입력 | R1 결론, R2 결론 `O((V + E) log V)` |
| `decrease-key-cost-missing` | 필수 전제 삭제 (키 감소 비용 없음) | 다익스트라 R2 만 보류. R1 과 독립된 BFS 결론은 남는다 |
| `third-note-duplicate` | 제3 문서에 같은 결론 | known (새 결론으로 내지 않음) |
| `plain-heap-no-decrease-key` | 자료 구조가 연산을 제공하지 않음 | R1 기각 |
| `heap-contradiction` | 노트끼리 모순 | 이진 힙 R1 기각, 피보나치 힙 결론은 그대로 |
| `stack-negative` | 음성 (스택은 우선순위 큐가 아님) | 결론 없음 |
| `bfs-linked-queue` | 다른 알고리즘 | R2 `O(V + E)` |
| `dijkstra-fibonacci-amortized` | 상각 비용 | R2 `O(V log V + E)`, 상각 |
| `wrong-premise-sorted-array` | 노트 전제가 틀림 (정렬 배열 삽입 O(1)) | R1 결론, R2 는 soundness 판정에서 기각 (`--sound` 일 때만 돈다) |

```bash
uv run ai derive                   # 사람이 적은 사실 + 규칙만 (LLM 없음, 같은 입력이면 같은 결과)
uv run ai derive --auto --sound    # 사실을 LLM 으로 뽑고 결론을 판정 (ANTHROPIC_API_KEY 필요)
```

**사례를 더할 때.** 노트를 `notes/` 에, 사람이 읽은 사실을 `facts/` 에 넣고 `cases.json` 에 기대 결과를 적는다.
기대 결과는 시스템 출력을 보기 전에 교재·공식 문서로 확인해 적는다. 최종 시험용 사례는 `"split": "test"` 로 두고,
규칙을 고칠 때는 `--split dev` 만 본다 (안건 6: 최종 테스트를 보고 규칙을 고치지 않는다).
지금 사례는 모두 dev 다. 엔진을 만들며 함께 쓴 것이라 독립된 시험이 아니다.
