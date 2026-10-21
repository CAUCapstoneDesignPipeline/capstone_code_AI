# BE 연동 가이드: 노트 연결과 새 지식 제안

상태: 구현됨 (AI 서버). 계약 정본은 `docs` 저장소의 [api/ai-api.md](../../docs/api/ai-api.md) ·
[api/ai-openapi.yaml](../../docs/api/ai-openapi.yaml) 이다. 이 문서는 **어떤 순서로 무엇을 부르는지**와
AI 쪽 구현이 정한 것(profile·판정 기준)을 적는다. 실행되는 참조 구현은 [`app/client.py`](../app/client.py) 다.

서버를 띄우면 실제 스키마는 `GET /internal/v1/openapi.json`, 화면은 `/internal/v1/docs` 에 있다.

## 1. 전체 흐름

```
노트 저장 ─▶ ① POST /extract            개념·관계·근거 (노트마다)
           ② POST /embeddings          개념 벡터 → BE 가 cosine 으로 병합 후보 쌍 검색
           ③ POST /merge-decisions     후보 쌍이 같은 개념인가 → BE 가 병합
           ④ 경로 찾기                  A —(노트 1 관계)→ B —(노트 2 관계)→ C
              · BE 가 자기 그래프에서 찾거나 (계약 F-PATH-01)
              · POST /note-links 로 AI 쪽 구현을 쓴다 (제안 operation, 아래 3장)
           ⑤ POST /verifications/bridges    B 가 두 노트에서 정말 같은 개념인가
           ⑥ POST /verifications/edges      A→B, B→C 가 근거에 정말 쓰여 있는가
           ⑦ POST /explanations             근거가 달린 설명 = 새 지식 제안 (A —합성 유형→ C)
           ⑧ POST /verifications/explanations 설명의 주장별 판정
           ⑨ BE 가 판정을 결합해 후보 / 가설 / 제외 (2장 표)
```

**권장 순서 (속도·비용·품질):** 경로 후보를 넉넉히(예: 12개) 모은 뒤 ⑤⑥ 을 먼저 묶어서 동시에 부른다
(⑤ 는 쌍 20개, ⑥ 은 간선 6개 정도씩 나눠 동시에. 같은 간선은 한 번만). 매개가 `different` 이거나 간선이 `contradicted` 인 경로는
여기서 버리고, 남은 경로를 "합성 규칙 있음 → 확인된 간선 수 → 매개 같음 → 연결 점수" 순으로 골라 ⑦⑧ 을 동시에 부른다.
LLM 설명(⑦)은 가장 비싸므로 살아남은 경로에만 쓴다. `app/client.py` `suggest_knowledge` 가 이 순서다.
측정 (노트 2개, 캐시 없음): 차례대로 부르던 때 59초 → 이 순서로 약 9초, 후보 0개 → 6개.
경로 고르기는 합성 규칙이 있는 경로를 먼저, 연결 노트마다 돌아가며 고른다 (한 노트의 경로로 다 채우지 않게).

## 2. 판정 결합 (계약 §5 표를 코드로 옮긴 것, `app/client.py` `decide`)

| 조건 | 결과 |
| --- | --- |
| 매개 `different` · 간선 `contradicted` · 설명 검증 `fail` 중 하나라도 | **제외** |
| 그 밖에 매개 `uncertain` · 간선 `unknown` · 설명 `uncertain`/gap · `proposedTypeCode = null` 중 하나라도 | **가설** |
| 모두 통과 | **후보** (사용자 검토 대상) |

## 3. 연결·경로 후보: `POST /internal/v1/note-links` (`findNoteLinks`)

> ⚠️ **계약 v0.1.0 에 없는 AI 쪽 제안 operation.** 계약상 경로·순위는 BE 책임이다. 그대로 쓸지, BE 가 같은 방법을
> 구현할지, 계약에 넣을지 팀이 정한다. 헤더·오류·멱등 규칙은 다른 operation 과 같다.

무상태다. 트리거 노트와 **같은 사용자의 후보 노트**를 보내면 각 노트에서 개념·관계를 뽑아 잇는다.

| 요청 필드 | 의미 |
| --- | --- |
| `context`, `profileVersion` (`links-det-1`) | 공통 |
| `note` | 트리거 `NoteSnapshot` (context 의 noteId·version 과 같아야 함) |
| `candidates` | 후보 `NoteSnapshot` 1~200개. 본문 합계가 요청 한도(2 MiB)를 넘지 않게 BE 가 고른다 (예: 같은 주제, 최근 수정, 개념 표기가 겹치는 노트) |
| `policy` | 관계 유형·합성 규칙. 경로의 `proposedTypeCode` 를 여기서 계산한다 |
| `k` (1~30, 기본 10), `minConfidence` (0~1, 기본 0) | 돌려줄 노트 수, 연결에 쓸 개념의 최소 신뢰도 (`/extract` 의 concept confidence 와 같은 값, 양쪽 노트 모두에 적용) |

응답 `links[]`: `noteId`, `score`, `bridges[]` (공유 개념, 양쪽 신뢰도, 양쪽 근거 문장 span),
`paths[]` (`source`, `bridge`, `target`, `steps[2]`, `proposedTypeCode`). 모든 span 은 원문 UTF-16 위치이고
quote 는 원문 slice 와 같다. **경로 단계의 근거 span 은 그 노트의 `/extract` 응답 evidence 와 같은 위치**이므로
BE 는 `(noteId, startOffset, endOffset)` 로 자기가 저장한 evidence·relation UUID 를 찾아 ⑤~⑧ 에 넣는다.

점수: 공유 개념마다 `양쪽 중요도의 곱 ÷ log(1 + 그 개념이 나온 노트 수)` 의 합 (Adamic-Adar). 후보 노트의 15% 와
5개 중 큰 수보다 많은 노트에 나오는 흔한 개념은 뺀다. 같은 개념의 판단은 표기 정규화(공백·하이픈·대소문자)와
**그 노트 안에서** 확인된 별칭(`교착 상태 (Deadlock)`, 약어 ↔ 풀어 쓴 말)만 쓴다.

## 4. profile

| operation | profileVersion | 방법 |
| --- | --- | --- |
| extractKnowledge | `extract-det-3` | Kiwi 명사구 + TF-IDF + 마크다운 구조. 관계는 한국어 문형 규칙 + 구조(제목·목록·정의·비교 표) |
| decideConceptMerges | `merge-det-1` | 아래 동일성 규칙 (병합용, 덜 엄격) |
| verifyBridgeIdentity | `bridges-det-1` | 아래 동일성 규칙 (매개 검증용, 엄격) |
| verifyEdgeEntailment | **`edges-llm-1`** (권장) | LLM 이 근거 원문만 읽고 entailed / contradicted / unknown. 관계 판정과 다른 프롬프트 (독립 검증). 간선마다 정확히 하나 아니면 502. 캐시 |
| verifyEdgeEntailment | `edges-det-1` | 두 개념이 근거에 나오는가 + 유형별 단서 말 + 부정 표현. 단서 말 없이 쓰인 관계(LLM 이 찾은 관계)는 대부분 unknown |
| generateExplanation | `explain-llm-6` | 항상 LLM(기본 `claude-opus-5-5`, effort medium)이 전제·조건·결론(새 지식)을 쓴다. 문체는 트리거 노트(context.noteId)의 문장 몇 개를 예시로 줘서 그 사람의 말투를 따른다. 경로·근거·합성 규칙은 입력으로 고정. 인용은 요청 근거 중에서만 고르게 구조화 출력 enum 으로 묶고, 응답은 코드가 §3.9 로 다시 검사한다 (어기면 502, 자동 재시도·모델 fallback 없음). 같은 입력은 캐시 |
| verifyExplanation | `explanation-det-2` | 주장마다 인용이 어느 단계 근거인지, 두 개념이 원문에 나오는지, 결론이 두 전제를 잇는지, 합성 규칙이 있는지 |
| findNoteLinks (제안) | **`links-hybrid-1`** (권장) | 3장 + 관계를 LLM 으로 보탠다 (`extract-hybrid-2` 의 관계와 같다) |
| findNoteLinks (제안) | `links-det-1` | 3장. 규칙·구조 관계만 써서 경로가 거의 없다 |

`explain-llm-6`, `edges-llm-1`, `links-hybrid-1`·`extract-hybrid-*` 의 관계 판정이 LLM 이고 나머지는 결정적이다. 판정 이유(reason)는 ~습니다체다. 검증 profile 은 모두 결정적이라 LLM 생성과 모델이 다르다 (계약 §5 독립성).

`meta.modelRevision` 은 그 방법 코드의 hash 다 (LLM profile 은 `모델 id@코드 hash`, `meta.model` 은 실제 호출한 모델 id). 코드가 바뀌면 값이 바뀌므로 BE 는 결과를 구분할 수 있다.
검증 profile 은 `generationProvenance` 의 각 항목과 모델 또는 prompt 가 달라야 한다 (같으면 422 `UNSUPPORTED_PROFILE`).
`/health/ready` 의 `profileVersions` 에 지금 받을 수 있는 profile 이 모두 나온다.

**동일성 규칙** (`app/services/reasoning/identity.py`): 표기가 같거나 원문이 둘을 같은 것으로 적었을 때만 같을 수 있다.
그다음 두 노트가 함께 쓰는 다른 명사구의 비율(작은 쪽 기준)로 문맥을 본다. 0.12 이상(3개 이상)이면 같음, 0.06 이상이면
병합에서는 같음·매개 검증에서는 같은 주제일 때만 같음, 그 밖은 애매. `DP` 같은 짧은 약어는 문맥이 충분할 때만 같음.
표기가 다르면 다름, 단 임베딩 cosine ≥ 0.92 이면 애매. 노트 묶음과 무관하게 같은 규칙이다 (IDF 를 쓰지 않는다).

## 5. 참조 구현으로 직접 해 보기

```python
import asyncio
from app.client import AiClient, snapshot, suggest_knowledge

async def main():
    client = AiClient("http://127.0.0.1:8000", token="<AI_SERVICE_TOKEN>")
    a = snapshot("…uuid…", "TCP", open("TCP.md").read())
    b = snapshot("…uuid…", "전송 계층", open("전송 계층.md").read())
    policy = {"version": "p1", "relationTypes": [{"code": "is_a", "label": "종류", "description": "…"}],
              "compositionRules": [{"firstType": "is_a", "secondType": "is_a", "resultType": "is_a"}]}
    res = await suggest_knowledge(client, a, [b], policy)
    for s in res["suggestions"]:
        print(s["status"], s["knowledge"])   # candidate 합성 규칙(is_a + is_a → is_a)에 따라 TCP는 네트워크 프로토콜의 한 종류다.

asyncio.run(main())
```

화면으로는 `uv run python run_playground.py [--notes 폴더]` → **지식 제안** 버튼. 지식 제안은 항상 LLM 이 쓰므로 `.env` 에
`ANTHROPIC_API_KEY` 가 있어야 한다 (설정: `AI_EXPLAIN_LLM_PROVIDER` / `_MODEL` / `_EFFORT`). 노트 폴더는 아무 `.md`/`.txt` 폴더나 된다.

## 6. 알려진 한계

- **제안 수는 관계 추출에 달려 있다.** 경로는 한 노트의 관계(A→B)와 다른 노트의 관계(B→C)가 같은 개념으로 이어질 때만
  생긴다. 규칙만으로는 관계가 적다 (DB/OS 노트 17개에서 관계 14개·경로 0개). 관계를 LLM 으로 보태면(`extract-hybrid-2`,
  `links-hybrid-1`) 같은 노트에서 관계 178개·경로 71개, 17개 중 15개 노트에 제안이 생겼다. LLM 이 찾은 관계는 표본에서
  약 3분의 2가 맞았고(농담·제목을 관계로 읽는 경우가 있다), 틀린 경로는 매개·간선·설명 검증이 가설이나 제외로 내린다.
- **합성 규칙은 BE 정책이다.** playground 의 규칙(`app/dev/router.py` `EXAMPLE_RULES`)은 흐름을 보기 위한 예시다.
- **IDF 는 개념 점수의 기준 통계다.** 배포에서는 고정 파일(`AI_CONCEPT_IDF`)을 쓴다. 다른 분야 노트가 많아지면 넓은
  참고 문서로 다시 만들고 새 profile 로 올린다. 동일성·간선·설명 판정은 IDF 를 쓰지 않는다.
- `candidate_claim` 의 근거가 하나뿐인 BE 테이블 문제(계약 §5 마지막 문단)는 그대로다. 결론은 근거가 2개 이상이다.
