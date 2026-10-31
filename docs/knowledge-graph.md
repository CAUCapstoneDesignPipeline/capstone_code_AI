# 지식 그래프의 키워드와 개념

상태: 정의만 있다 (코드: [`app/schemas/knowledge.py`](../app/schemas/knowledge.py), 시험: `tests/schemas/test_knowledge.py`).
정본은 팀 문서의 [회의록 10/30 지식 그래프 설계](../../docs/meetings/2026-10-30-knowledge-graph.md) 이고, 이 문서는 그 구조를 AI 쪽에서
어떤 필드와 검사로 옮겼는지만 적는다. 아직 API 계약에는 없다 (BE 와 맞춘 뒤 `ai-openapi.yaml` 에 올린다).

## 구조

```text
Keyword (노드)                       키워드 하나 = 노드 하나. 같은 뜻의 개념을 묶는다
  Concept 1: 설명 + 소스(페이지 구간)  키워드 안의 설명 1개
  Concept 2: 설명 + 소스
  ...
```

```json
{
  "keywordId": "…uuid", "name": "힙", "aliases": ["heap"],
  "concepts": [
    {"conceptId": "…uuid", "origin": "extracted",
     "explanation": "힙은 부모가 자식보다 작은 완전 이진 트리예요.",
     "sources": [{"noteId": "…", "noteVersion": 3, "span": {"startOffset": 120, "endOffset": 152, "quote": "…"}}]},
    {"conceptId": "…uuid", "origin": "generated",
     "explanation": "…",
     "sources": [{"noteId": "…A"}, {"noteId": "…B"}],
     "external": [{"url": "https://…", "title": "…", "retrievedAt": "2026-10-31T09:00:00Z"}]}
  ]
}
```

(`generated` 의 `sources` 는 줄여 썼다. 실제로는 첫 개념처럼 `noteVersion`·`span` 이 모두 들어간다.)

| 필드 | 뜻 |
| --- | --- |
| `Keyword.name` / `aliases` | 보여 줄 이름과 같은 뜻의 다른 표기 |
| `Keyword.concepts` | 1개 이상. 개념 없는 키워드는 만들지 않는다 |
| `Concept.origin` | `extracted` 노트에서 뽑은 개념 · `generated` 사용자가 모르는 지식으로 만든 개념 |
| `Concept.explanation` | 설명 1개 (최대 2000자, 임베딩 입력 한도와 같다) |
| `Concept.sources` | 개념이 나온 페이지(노트)와 구간. 구간은 기존 `Span` (원문 인용 포함) |
| `Concept.external` | 생성에 쓴 인터넷 자료 (URL·제목·가져온 시각·인용). 결과 카드에 URL 을 보이려고 남긴다 |

## 검사하는 것

- `extracted`: 소스는 페이지 하나, 외부 자료 없음.
- `generated`: 소스에 도출에 쓴 개념들의 페이지를 모두 넣고, **서로 다른 노트가 둘 이상**이어야 한다 (기능 명세 1.4 "복수 문서의 실질적 기여").
- 한 키워드 안에서 `conceptId` 는 겹치지 않고, `name` 이 `aliases` 에 또 들어 있지 않다.
- URL 은 `http(s)` 만 받는다. 모든 객체는 모르는 필드를 거절한다 (`Closed`).

여기서 확인하지 않는 것: 같은 뜻인지 (병합은 `/embeddings` + `/merge-decisions` 몫), 인용이 노트 원문과 일치하는지
(기존 `/extract` 의 구간 검사 몫), 설명이 근거에서 나오는지 (판정 몫).

## 정하지 않은 것과 모델이 남겨 둔 자리

| 미정 (회의록) | 모델에서 |
| --- | --- |
| 개념의 `etc` 필드 | 두지 않았다. 파이프라인이 필요로 하는 것이 생기면 그때 한 필드씩 추가한다 |
| 소스에 인터넷 URL 을 넣을지 | `sources` (페이지) 와 `external` (인터넷) 을 나눠 두었다. 넣기로 하면 합치면 되고, 안 넣기로 해도 결과 카드용 URL 은 남는다 |
| 결과가 새 키워드인지, 기존 키워드의 새 개념인지 | 둘 다 표현된다: `generated` 개념을 새 `Keyword` 에 넣거나 기존 `Keyword.concepts` 에 더하면 된다 |
| 거절한 조합의 기록 | 이 모델 밖이다 (개념 쌍과 시각을 따로 저장) |

## 쓰임

- 사용자용 그래프의 간선: 같은 키워드에 개념을 넣은 두 페이지를 잇는다. `Keyword.note_ids()` 가 그 페이지 집합이다.
- 기능용 그래프의 간선은 키워드 사이 "관계가 있다" 정도이고 유형은 쓰지 않을 가능성이 높다 (회의록). 이 문서는 노드만 정의한다.
- 새 지식 생성의 입력은 키워드가 아니라 그 안의 개념 P·Q 다. 생성된 개념의 `sources` 에는 P·Q 의 페이지가 반드시 들어간다.
