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
           ⑦ POST /explanations             근거가 달린 설명 = 새 지식 제안 (A —합성 유형→ C).
                                            단계 근거 + 문맥 근거를 보낸다 (아래 "문맥 근거")
           ⑧ POST /verifications/explanations 설명의 주장별 판정 (근거 뒷받침·구체성·새로움)
           ⑨ BE 가 판정을 결합해 후보 / 가설 / 제외 (2장 표). 후보만 사용자에게 보인다 (가설·제외는 이유만 저장)
```

**권장 순서 (속도·비용·품질):** 경로 후보를 넉넉히(예: 12개) 모은 뒤 ⑤⑥ 을 먼저 묶어서 동시에 부른다
(⑤ 는 쌍 20개, ⑥ 은 간선 6개 정도씩 나눠 동시에. 같은 간선은 한 번만). 매개가 `different` 이거나 간선이 `contradicted` 인 경로는
여기서 버린다. 남은 경로는 `app/client.py` `pick_paths` 처럼 고른다:
1. **같은 A–C 는 하나로.** 표기(대소문자·띄어쓰기)를 무시하고 시작·끝 개념이 같은 경로는 같은 지식이다. 가장 나은 경로를 남기고
   다른 경로의 같은 단계 근거를 그 단계 근거에 보탠다 (기능 명세 1.4 "기존 결과와의 중복"; 저장된 관계와의 중복 확인은 BE 몫이다).
2. **가치 순:** 합성 규칙 있음 → 분류 계층만 올라가는 경로(`is_a`·`example_of` 만)가 아님 → 확인된 간선 수 → 매개 같음 → 연결 점수.
3. **고르게:** 이미 고른 경로와 첫 단계(A→B)나 끝 단계(B→C)가 같은 경로는 다른 경로를 먼저 고른 뒤 남는 자리에 넣는다.

고른 경로만 ⑦⑧ 을 동시에 부른다. LLM 설명(⑦)은 가장 비싸므로 살아남은 경로에만 쓴다. `suggest_knowledge` 가 이 순서다.
측정 (노트 2개, 캐시 없음): 차례대로 부르던 때 59초 → 이 순서로 약 9초, 후보 0개 → 6개.
경로 고르기는 합성 규칙이 있는 경로를 먼저, 연결 노트마다 돌아가며 고른다 (한 노트의 경로로 다 채우지 않게).

## 2. 판정 결합 (계약 §5 표를 코드로 옮긴 것, `app/client.py` `decide`)

| 조건 | 결과 |
| --- | --- |
| 매개 `different` · 간선 `contradicted` · 설명 검증 `fail` 중 하나라도 | **제외** |
| 그 밖에 매개 `uncertain` · 간선 `unknown` · 설명 `uncertain`/gap · `proposedTypeCode = null` 중 하나라도 | **가설** |
| 모두 통과 | **후보** (사용자 검토 대상) |

**가설은 화면에 내지 않는다** (팀 회의 2026-10-29: 도출 조건이 부족하면 결과에서 뺀다). `suggest_knowledge` 는 후보만
`suggestions` 에 넣고, 가설은 `screened.held`, 제외는 `screened.rejected` 에 이유와 함께 남긴다 (평가·디버깅용).

**문맥 근거.** 두 관계만 주면 설명은 "A 는 C 다" 에서 끝난다. ⑦⑧ 의 `evidence` 에 단계 근거 말고도 **단계 노트의 다른 구간**을
넣으면 AI 는 그것을 문맥 근거로 본다: 설명은 결론에 쓰는 사실(C 의 성질·규칙)을 그 구간을 인용한 보조 전제로 적고, 결론은
그것을 A 에 적용한 결과를 쓴다. 모든 문장이 인용 근거를 가지므로 ⑧ 이 확인할 수 있다. `/note-links` 경로 단계의 `context`
(같은 절에서 그 단계의 개념을 말하는 문장, 최대 6개)를 그대로 넣으면 된다. 요청 근거 한도(40개) 안에서 단계마다 4개 정도면 충분하다.
문맥 근거는 단계 노트에 있어야 한다 (다른 노트의 구간은 경로 밖 근거다).

**새 지식의 조건 (기능 명세 1.4) 과 확인하는 곳**

| 조건 | 어디서 |
| --- | --- |
| 기존 내용의 요약·바꿔쓰기가 아님 | ⑦ 프롬프트 + ⑧ `explanation-hybrid-2` 의 novel 판정 (아니면 결론 fail → 제외) |
| 둘 이상 문서의 실질적 기여 | 경로 (두 단계가 다른 노트, `/explanations` 가 같은 노트 경로를 422 로 거절) |
| 명확한 관계 또는 결론 | ⑧ `explanation-hybrid-2` 의 specific 판정 (아니면 결론 fail → 제외) |
| 원문 근거 | 모든 주장이 요청 근거를 인용 (enum), ⑧ 이 주장마다 근거가 뒷받침하는지 판정 |
| 추론 과정 명시 | 전제·보조 전제·조건·가정·결론으로 나눈 claims, ⑧ 의 gaps |
| 기존 결과와의 중복 | 한 번의 제안 안: 같은 A–C 합치기 (위). 저장된 관계와의 중복: BE (F-CAN-02) |

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
`paths[]` (`source`, `bridge`, `target`, `steps[2]`, `proposedTypeCode`). 단계마다 `evidence` (관계 근거) 와 `context`
(같은 절에서 그 단계의 두 개념 중 하나 이상을 말하는 다른 문장, ⑦⑧ 의 문맥 근거). 모든 span 은 원문 UTF-16 위치이고
quote 는 원문 slice 와 같다. **경로 단계의 근거 span 은 그 노트의 `/extract` 응답 evidence 와 같은 위치**이므로
BE 는 `(noteId, startOffset, endOffset)` 로 자기가 저장한 evidence·relation UUID 를 찾아 ⑤~⑧ 에 넣는다.

점수: 공유 개념마다 `양쪽 중요도의 곱 ÷ log(1 + 그 개념이 나온 노트 수)` 의 합 (Adamic-Adar). 후보 노트의 15% 와
5개 중 큰 수보다 많은 노트에 나오는 흔한 개념은 뺀다. 같은 개념의 판단은 표기 정규화(공백·하이픈·대소문자)와
**그 노트 안에서** 확인된 별칭(`교착 상태 (Deadlock)`, 약어 ↔ 풀어 쓴 말)만 쓴다.

## 4. 조건부 결론: `POST /facts` · `POST /derivations` (R1 구현 적용 · R2 비용 합성)

계약 v0.1.0 에 없는 AI 쪽 제안 operation 이다. A–B–C 관계 합성과 별개로, 노트 문장을 **계산용 사실**로 구조화한 뒤
규칙으로 결론을 낸다. 예: 다익스트라 노트("삽입 V번, 최솟값 추출 V번, 키 감소 최대 E번, 큐 크기 ≤ V, 큐 밖 O(V+E)") +
인덱스 있는 이진 최소 힙 노트("세 연산 모두 O(log n)") → **"인덱스 있는 이진 최소 힙으로 구현한 다익스트라의 시간은 O((V + E) log V) 이하예요"**.

```
노트 저장 ─▶ ① POST /facts          노트 하나 → 계산용 사실 (facts-llm-1). (noteId, noteVersion) 마다 한 번, BE 가 저장
분석 실행 ─▶ ② 관련 노트 고르기      트리거 노트와 역할·연산·주체가 겹치는 노트 (최대 20개, app/client.py related_notes)
           ③ POST /derivations    그 노트들의 사실 → 결론 (derive-sound-1)
           ④ status = conclusion 만 화면에. held·rejected·known 은 이유만 저장
```

**사실 (`/facts` 응답, `/derivations` 요청).** 사실 하나 = 종류 + 주체 + 원문 근거 하나.

| kind | 뜻 | 주요 칸 |
| --- | --- | --- |
| `needs_op` | 알고리즘이 역할에서 연산을 쓴다 | `role`, `op`, `expr` = 최대 횟수 (없으면 횟수 모름) |
| `uses` | 알고리즘이 역할을 쓴다 (연산은 말하지 않음) | `role` |
| `size_bound` | 역할(자료 구조)의 크기 n ≤ expr | `role`, `expr` |
| `extra_cost` | 역할 연산 밖 비용 | `role`, `expr` |
| `implements` | 자료 구조가 역할의 구현이다 | `role`, `negated` |
| `op_cost` | 자료 구조가 연산을 제공한다, 비용 expr (n = 자기 크기) | `op`, `expr`, `case` = worst/amortized/average, `negated` = 제공하지 않음 |
| `requires` | 성립 조건 (비음수 가중치, 비교 가능한 키) | `condition` |
| `stated_result` | 노트가 이미 쓴 결론 (알고리즘을 자료 구조로 구현하면 expr) | `other`, `otherKey`, `expr` |

역할·연산은 닫힌 어휘다 (`app/services/derive/facts.py` `ROLES`·`OPS`). 주체는 `subjectKey`(영어 소문자·밑줄)로 노트 사이를
잇는다. `/facts` 에 `knownSubjects`(사용자의 기존 subjectKey 들)를 주면 같은 대상에 같은 key 를 쓴다.
`origin` 은 `extracted`(LLM) | `manual`(사람이 적음). 사람이 보완한 사실로 낸 결론은 `automation` 이 `manual`/`mixed` 라서 자동 추출 성능과 따로 잰다.

**근거 검사.** `/derivations` 는 사실마다 `evidence` 의 noteVersion·bodySha256·UTF-16 위치·quote 가 요청 `documents` 원문과
맞는지 본다. 옛 리비전에서 뽑은 사실은 422 (`INVALID_REFERENCE`/`INVALID_SPAN`). 그래서 늦게 온 옛 응답이 최신 원문으로 결론을 만들 수 없다.
`/facts` 는 노트에 글자 그대로 있는 quote 만 사실로 남긴다 (나머지는 `dropped` 에 이유별 개수).

**결론 (`derivations[]`).**

| 칸 | 뜻 |
| --- | --- |
| `rule` · `ruleVersion` | `R1` (`r1-implementation-1`) · `R2` (`r2-cost-1`) |
| `status` | `conclusion` 보여 줄 결론 / `held` 필수 전제 없음 (`missing`) / `rejected` 전제 모순·연산 미제공·soundness 실패 / `known` 이미 있는 결론 |
| `conclusion` · `result` · `case` | 결론 문장 (해요체), R2 의 `O(...)`, 최악·상각·평균 |
| `conditions` | 결론이 성립하는 조건 (노트의 `requires` + "상한이에요, 가장 빠르다는 뜻이 아니에요") |
| `steps` | 계산 과정. 단계마다 쓴 사실 id 와 계산한 식 |
| `premiseFactIds` · `noteIds` | 전제 사실, 전제를 낸 노트. **이 노트의 원문이 바뀌면 BE 는 결론을 재검토 상태로** |
| `dependsOn` | R2 가 기대는 R1 결론 id |
| `duplicateOfFactIds` | `known` 일 때 이미 같은 결론을 쓴 사실 |
| `explanation` | 사용자에게 보일 설명 (단계마다 어느 노트에서 온 사실인지) |
| `soundness` | `derive-sound-1`: LLM 판정 (sound/unsound/uncertain, 충실하지 않은 전제) |

`noveltyScope.noteIds` 는 "이미 있는 결론인가" 를 찾아본 노트(요청 문서)다. 그 밖의 노트는 보지 않았으므로 화면 문구는
"찾아본 노트에서는 없었어요" 다. 사용자 전체 노트에서 찾으려면 BE 가 저장된 `stated_result` 사실 중 같은 `subjectKey` 를 가진
노트도 요청에 넣는다.

**판정 규칙 (결정적).**
- R1: 알고리즘이 역할에서 쓰는 연산을 자료 구조가 모두 제공하면 구현할 수 있다. 연산이 하나라도 노트에 없으면 held,
  "제공하지 않는다" 고 적혀 있으면 rejected, 노트끼리 다르게 말하면 rejected.
- R2: R1 + 연산 횟수·연산 비용·크기 상한·역할 밖 비용이 모두 있으면 Σ 횟수 × 비용[n := 크기 상한] + 밖 비용 을 정리한다
  (계수·작은 항 버림, 빼기는 상한이라 버림). 하나라도 없거나 식을 읽지 못하면 held. **R2 가 보류여도 R1 은 남는다.**
- 두 노트의 기여: 필수 전제가 한 노트에 모두 있으면 known (새 결론이 아니다). 노트가 같은 결론을 이미 썼으면 known.
- 모델 상식·웹으로 빠진 전제를 채우지 않는다. 생성된 결론은 사실로 되돌려 넣지 않는다.
- `derive-sound-1` 은 conclusion 마다 LLM 이 전제의 원문 충실성과 결론의 soundness 를 판정한다: 충실하지 않음·unsound → rejected,
  uncertain → held. R1 이 떨어지면 그것에 기대는 R2 도 떨어진다. 판정만 하고 전제를 채우지 않는다.

**보여 줄 카드** (`app/client.py` `pick_insights`): (알고리즘, 자료 구조, 역할) 마다 하나. R2 결론이 있으면 R2 카드(R1 은 그 안의 단계),
없으면 R1. 트리거 노트가 전제를 낸 결론만, 최대 5개.

## 5. profile

| operation | profileVersion | 방법 |
| --- | --- | --- |
| extractKnowledge | `extract-det-3` | Kiwi 명사구 + TF-IDF + 마크다운 구조. 관계는 한국어 문형 규칙 + 구조(제목·목록·정의·비교 표) |
| extractKnowledge | **`extract-hybrid-3`** (권장) | `extract-det-3` 의 개념 후보를 LLM(관계 LLM, effort low)이 노트 원문과 함께 읽고 개념이 아닌 것(일반 낱말·문장 조각·예시 값·이름)을 **빼기만** 한다. 위치·표기는 결정적 추출 그대로. 관계는 `extract-hybrid-2` 와 같다. 노트 원문으로 캐시. 처음 보는 노트에서 정밀도 43% → 73%, 재현율 85% → 84% ([측정](eval/links-and-explanations.md)) |
| decideConceptMerges | `merge-det-1` | 아래 동일성 규칙 (병합용, 덜 엄격) |
| decideConceptMerges | `merge-hybrid-1` | `merge-det-1` + 규칙이 same 이라 하지 못한 쌍과 주제가 다른 same 쌍을 LLM(검증 LLM, effort low)이 두 표기와 근거 문장만 보고 판정. LLM same 은 확신 0.8 이상만 같음, 규칙 same 인데 LLM 이 아니라고 하면 애매. 쌍마다 캐시. 번역·음역(`Mutex ↔ 뮤텍스`)을 잇는다 ([측정](eval/alias-embeddings.md)) |
| verifyBridgeIdentity | `bridges-det-1` | 아래 동일성 규칙 (매개 검증용, 엄격) |
| verifyEdgeEntailment | **`edges-llm-2`** (권장) | `edges-llm-1` + 근거마다 노트의 제목 경로("DB > WAL > 오해")를 보여 준다. '오해'·'예전 방식' 절의 문장을 글쓴이의 주장으로 읽지 않게 |
| verifyEdgeEntailment | `edges-llm-1` | LLM 이 근거 원문만 읽고 entailed / contradicted / unknown. 관계 판정과 다른 프롬프트 (독립 검증). 간선마다 정확히 하나 아니면 502. 캐시 |
| verifyEdgeEntailment | `edges-det-1` | 두 개념이 근거에 나오는가 + 유형별 단서 말 + 부정 표현. 단서 말 없이 쓰인 관계(LLM 이 찾은 관계)는 대부분 unknown |
| generateExplanation | `explain-llm-8` | 항상 LLM(기본 `claude-opus-5-5`, effort medium)이 전제·보조 전제(문맥 근거)·조건·결론(새 지식)을 쓴다. 결론은 A–C 관계와 그것을 A 에 적용한 결과 2~4 문장, 근거보다 넓게 말하지 않는다. 문체는 해요체(~에요/~예요), 용어 쓰는 방식은 트리거 노트(context.noteId)를 따른다. 경로·근거·합성 규칙은 입력으로 고정. 인용은 요청 근거 중에서만 고르게 구조화 출력 enum 으로 묶고, 응답은 코드가 §3.9 로 다시 검사한다 (어기면 502, 자동 재시도·모델 fallback 없음). 같은 입력은 캐시 |
| verifyExplanation | **`explanation-hybrid-2`** (권장) | `explanation-det-3` + LLM(검증 LLM, effort low)이 주장마다 인용 근거가 뒷받침하는지(supported → pass, partial → uncertain, unsupported·contradicted → fail), 결론이 구체적인지·새로운지 본다. 구체적이지 않거나 한 노트 내용의 바꿔쓰기면 결론 fail. 생성기와 prompt 가 달라 독립. 주장마다 정확히 하나 아니면 502. 캐시 |
| verifyExplanation | `explanation-det-3` | `explanation-det-2` + 문맥 근거만 인용하는 보조 전제를 받는다 (주장과 인용 원문이 같은 경로 개념을 말하면 통과) |
| verifyExplanation | `explanation-det-2` | 주장마다 인용이 어느 단계 근거인지, 두 개념이 원문에 나오는지, 결론이 두 전제를 잇는지, 합성 규칙이 있는지. 단계 근거가 아닌 인용은 경로 밖 근거로 본다 |
| findNoteLinks (제안) | **`links-hybrid-1`** (권장) | 3장 + 관계를 LLM 으로 보탠다 (`extract-hybrid-2` 의 관계와 같다) |
| findNoteLinks (제안) | `links-det-1` | 3장. 규칙·구조 관계만 써서 경로가 거의 없다 |
| extractFacts (제안) | `facts-llm-1` | 4장. LLM(사실 LLM, 기본 effort medium)이 노트에 쓰인 계산용 사실과 원문 구절을 고른다. quote 가 원문에 그대로 없거나 어휘 밖이면 버림. 거절이면 사실 없음. 노트 원문·knownSubjects 로 캐시 |
| deriveConclusions (제안) | `derive-rules-1` | 4장 R1·R2 규칙만 (결정적, LLM 없음) |
| deriveConclusions (제안) | **`derive-sound-1`** (권장) | `derive-rules-1` + 규칙을 통과한 결론을 LLM(soundness LLM, effort medium)이 판정. 사실 추출과 prompt 가 달라 독립 |

`explain-llm-8`, `edges-llm-*`, `explanation-hybrid-2`, `merge-hybrid-1`, `links-hybrid-1`·`extract-hybrid-*` 의 관계 판정이 LLM 이고 나머지는 결정적이다.
Anthropic 호출은 역할마다 고정인 system prompt 를 캐시한다 (같은 역할의 요청이 5분 안에 이어지면 system 을 캐시에서 읽는다).
새 지식(설명)은 해요체, 판정 이유(reason)는 ~습니다체다. LLM 검증 profile 은 생성 profile 과 prompt 가 달라 계약 §5 독립성을 지킨다.

`meta.modelRevision` 은 그 방법 코드의 hash 다 (LLM profile 은 `모델 id@코드 hash`, `meta.model` 은 실제 호출한 모델 id). 코드가 바뀌면 값이 바뀌므로 BE 는 결과를 구분할 수 있다.
검증 profile 은 `generationProvenance` 의 각 항목과 모델 또는 prompt 가 달라야 한다 (같으면 422 `UNSUPPORTED_PROFILE`).
`/health/ready` 의 `profileVersions` 에 지금 받을 수 있는 profile 이 모두 나온다.

**동일성 규칙** (`app/services/reasoning/identity.py`): 표기가 같거나 원문이 둘을 같은 것으로 적었을 때만 같을 수 있다.
그다음 두 노트가 함께 쓰는 다른 명사구의 비율(작은 쪽 기준)로 문맥을 본다. 0.12 이상(3개 이상)이면 같음, 0.06 이상이면
병합에서는 같음·매개 검증에서는 같은 주제일 때만 같음, 그 밖은 애매. `DP` 같은 짧은 약어는 문맥이 충분할 때만 같음.
표기가 다르면 다름, 단 임베딩 cosine ≥ 0.92 이면 애매. 노트 묶음과 무관하게 같은 규칙이다 (IDF 를 쓰지 않는다).

## 6. 참조 구현으로 직접 해 보기

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
        print(s["status"], s["knowledge"])   # candidate TCP는 전송 계층 프로토콜이고 … 그래서 TCP도 … 해요.
    print(res["screened"])                   # 경로 수, 합친 수, 검증에서 뺀 제안(rejected)과 이유

asyncio.run(main())
```

화면으로는 `uv run python run_playground.py [--notes 폴더]` → **지식 제안** 버튼. 지식 제안은 항상 LLM 이 쓰므로 `.env` 에
`ANTHROPIC_API_KEY` 가 있어야 한다 (설정: `AI_EXPLAIN_LLM_PROVIDER` / `_MODEL` / `_EFFORT`). 노트 폴더는 아무 `.md`/`.txt` 폴더나 된다.

조건부 결론은 `derive_insights(client, note, others, facts_cache)` (노트마다 `/facts` 한 번, `/derivations` 한 번).
사례로 시험: `uv run ai derive` (사람이 적은 사실, LLM 없음), `uv run ai derive --auto --sound` (LLM 추출 + 판정). 사례는 [docs/team/data](team/data/README.md).

## 7. 알려진 한계

- **제안 수는 관계 추출에 달려 있다.** 경로는 한 노트의 관계(A→B)와 다른 노트의 관계(B→C)가 같은 개념으로 이어질 때만
  생긴다. 규칙만으로는 관계가 적다 (DB/OS 노트 17개에서 관계 14개·경로 0개). 관계를 LLM 으로 보태면(`extract-hybrid-2`,
  `links-hybrid-1`) 같은 노트에서 관계 178개·경로 71개, 17개 중 15개 노트에 제안이 생겼다. LLM 이 찾은 관계는 표본에서
  약 3분의 2가 맞았고(농담·제목을 관계로 읽는 경우가 있다), 틀린 경로는 매개·간선·설명 검증이 가설이나 제외로 내린다.
- **합성 규칙은 BE 정책이다.** playground 의 규칙(`app/dev/router.py` `EXAMPLE_RULES`)은 흐름을 보기 위한 예시다.
- **IDF 는 개념 점수의 기준 통계다.** 배포에서는 고정 파일(`AI_CONCEPT_IDF`)을 쓴다. 다른 분야 노트가 많아지면 넓은
  참고 문서로 다시 만들고 새 profile 로 올린다. 동일성·간선·설명 판정은 IDF 를 쓰지 않는다.
- `candidate_claim` 의 근거가 하나뿐인 BE 테이블 문제(계약 §5 마지막 문단)는 그대로다. 결론은 근거가 2개 이상이다.
- **조건부 결론은 알고리즘·자료 구조의 시간 상한만 다룬다** (R1·R2). 역할·연산 어휘 밖의 내용(SQL, 확률, 수식)은 사실이 되지 않는다.
  R3(용량 비교)는 팀 결정으로 이번 범위 밖이다.
