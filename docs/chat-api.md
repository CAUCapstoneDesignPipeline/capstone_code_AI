# 챗봇 API: 새 지식 카드를 쉽게 풀어 주는 여러 턴 대화

상태: 구현됨 (AI 서버, 제안 operation). 아직 계약 문서(`docs` 저장소의 `api/ai-api.md`·`ai-openapi.yaml`)에는 없다. BE 와 맞춘 뒤 올린다.
회의록 10/30 "이해가 안 됨 → 챗봇으로 넘어간다. 근거 A, 근거 B, 외부 지식을 컨텍스트로 넘기고, 챗봇은 이를 쉽게 풀어 설명한다. 대화 응답은 BE·AI 가 맡는다" 를 구현한 것이다.
코드: [`app/api/v1/chat.py`](../app/api/v1/chat.py) · [`app/services/chat.py`](../app/services/chat.py) · [`app/schemas/chat.py`](../app/schemas/chat.py).

## 한눈에

- `POST /internal/v1/chat` 한 개. 기본은 **SSE 로 토큰마다** 답이 흘러오고, `"stream": false` 면 다 쓴 답 하나를 JSON 으로 준다.
- **여러 턴**: AI 서버는 대화를 저장하지 않는다. **BE 가 대화 기록을 보관하고 매 요청에 지금까지의 `messages` 를 모두 보낸다.** 카드(`knowledge`)도 매번 같이 보낸다.
- "이해가 안 돼요" 에는 앞서 한 설명과 카드를 함께 보고, 같은 말을 되풀이하지 않고 방식을 바꿔 (비유·단계별·예시) 다시 설명한다. 어느 부분이 어려운지 모르면 짧게 되묻는다.

## 요청

헤더는 다른 operation 과 같다: `Authorization: Bearer <서비스 토큰>`, `X-Request-Id: <uuid>`, `Content-Type: application/json`, `X-Input-SHA256: <본문 sha256>`, (선택) `traceparent`.
한 번 보내면 되돌릴 수 없는 스트림이라 `X-Attempt-Id`·`Idempotency-Key` 는 받지 않는다 (재시도는 BE 가 새 요청으로 한다).

```json
{
  "conversationId": "81738c89-df53-57b7-9a06-320a4deb50c8",
  "knowledge": {
    "newKeyword": "이중 버퍼링",
    "conclusion": "DB 와 OS 가 둘 다 페이지를 캐시해서 같은 데이터가 메모리에 두 번 올라갈 수 있어요.",
    "evidence": [
      {"label": "A", "keyword": "디스크", "note": "DB-WAL", "text": "DB는 buffer pool에 페이지를 올려놓고 거기서 수정한다."},
      {"label": "B", "keyword": "페이지 캐싱", "note": "OS", "text": "디스크에서 읽은 데이터를 RAM에 저장해 재사용하는 것을 페이지 캐싱이라고 한다."}
    ],
    "external": [{"url": "https://example.com/innodb", "title": "InnoDB", "quote": "O_DIRECT 는 OS 페이지 캐시를 거치지 않는다."}]
  },
  "messages": [
    {"role": "user", "content": "이중 버퍼링이 뭐예요?"},
    {"role": "assistant", "content": "DB 와 OS 가 같은 데이터를 두 번 캐시하는 거예요."},
    {"role": "user", "content": "이해가 안 돼요"}
  ],
  "stream": true
}
```

| 필드 | 규칙 |
| --- | --- |
| `conversationId` | UUID. 추적용이다. 서버는 이 값으로 대화를 찾지 않는다 |
| `knowledge` | 사용자가 보고 있는 새 지식 카드. `evidence` 1~4개 (각 `text` 2,000자 이하), `external` 0~5개 (`url` 은 `http(s)`) |
| `messages` | 1~40개, 각 4,000자 이하. **`user` 로 시작해 `user` 로 끝나고, `user` 와 `assistant` 가 번갈아** 나와야 한다 (아니면 400) |
| `stream` | 기본 `true` |

카드와 대화를 합친 글자 수가 60,000자를 넘으면 **자르지 않고** 413 `INPUT_TOO_LARGE` 로 거절한다. 오래된 턴을 줄이는 일은 BE 가 한다.
카드 안의 글(노트·웹에서 온 글)은 자료일 뿐 지시가 아니다 — 챗봇이 그 안의 지시를 따르지 않도록 system 프롬프트에서 못 박는다.

## 응답: SSE (`stream: true`)

`Content-Type: text/event-stream`. 이벤트마다 `event: 이름` 과 `data: JSON` 이고, 모두 `t`(요청을 받은 뒤 초)가 있다.

| 이벤트 | data | 뜻 |
| --- | --- | --- |
| `start` | `requestId`, `conversationId`, `model` | 시작 |
| `thinking` | – | 모델이 생각 중 (글 없음). 있을 수도 없을 수도 있다 |
| `delta` | `text` | 답의 다음 조각. 이어 붙이면 답이 된다 |
| `done` | `stopReason`, `model`, `usage{inputTokens, outputTokens, cacheReadTokens}` | 끝. `stopReason` 이 `max_tokens` 면 답이 길어서 끊긴 것이다 |
| `error` | `code`, `message`, `retryable` | 글이 나온 뒤의 오류 (`MODEL_TIMEOUT`, `MODEL_PROVIDER_ERROR`, `INVALID_MODEL_OUTPUT`). 이 이벤트로 끝난다 |

첫 글자가 나오기 전의 오류(제공자 인증·한도·거절 등)는 SSE 가 아니라 **보통의 계약 오류 응답** (JSON, HTTP 상태) 으로 낸다 (`AI_BUSY` 429, `SERVICE_UNAVAILABLE` 503, …). 그러므로 `200` 이면 스트림이 시작된 것이다.

```
event: start
data: {"requestId": "…", "conversationId": "…", "model": "claude-sonnet-5-5", "t": 1.63}

event: delta
data: {"text": "그럼 도서관 비유로 ", "t": 1.82}

event: delta
data: {"text": "다시 말해 볼게요.\n", "t": 1.84}

event: done
data: {"stopReason": "end_turn", "model": "claude-sonnet-5-5", "usage": {"inputTokens": 462, "outputTokens": 362, "cacheReadTokens": 1140}, "t": 4.77}
```

**연결을 끊으면 모델 호출도 멈춘다** (더는 토큰을 쓰지 않고 처리 자리도 돌려준다). 사용자가 "중단" 을 누르면 BE 가 이 연결을 닫으면 된다.

## 응답: JSON (`stream: false`)

```json
{"message": {"role": "assistant", "content": "…"}, "stopReason": "end_turn", "model": "claude-sonnet-5-5",
 "usage": {"inputTokens": 462, "outputTokens": 362, "cacheReadTokens": 1140}, "conversationId": "…", "elapsedMs": 4780}
```

## 오류

다른 operation 과 같은 계약 형식 (`{"error": {"code", "message", "retryable", "requestId"}}`).

| 상황 | 코드 (HTTP) |
| --- | --- |
| 토큰 없음·틀림 | `SERVICE_UNAUTHENTICATED` (401) |
| `X-Request-Id`·본문 형식·`messages` 모양 | `VALIDATION_FAILED` (400) |
| 본문 해시 불일치 | `INPUT_IDENTITY_MISMATCH` (409) |
| 한 메시지 4,000자·전체 60,000자 초과 | `INPUT_TOO_LARGE` (413) |
| 동시 처리 자리가 가득 참 | `AI_BUSY` (429, `Retry-After`) |
| 챗봇이 꺼져 있음 (`AI_CHAT_LLM` 이 `1` 이 아님) 또는 LLM 이 없음 (API 키 없음) | `SERVICE_UNAVAILABLE` (503) |
| 모델 제한 시간 (115초) | `MODEL_TIMEOUT` (504) / 스트림 중이면 `error` 이벤트 |
| 제공자 오류 | `MODEL_PROVIDER_ERROR` (502) |
| 모델이 거절 (재시도해도 같다) | `INVALID_MODEL_OUTPUT` (502, `retryable: false`) |

자동 재시도와 다른 모델로의 fallback 은 없다 (계약). 스트림 하나가 동시 처리 자리(`AI_MAX_INFLIGHT`) 하나를 끝날 때까지 쓴다.

## 화면(FE)이 정할 때 알아야 할 것 (회의 2026-11-08 질문에 대한 답)

| 질문 | 답 |
| --- | --- |
| 스트리밍인가 | 예. 기본이 SSE 토큰 스트림이다 (`delta` 를 이어 붙여 그린다). 한 번에 받고 싶으면 `"stream": false`. 첫 글자까지 약 2~3초, 끝까지 약 5초 (위 측정) |
| 중지를 둘 수 있나 | 예. 연결이 끊기면 응답 제너레이터가 닫히면서 모델 호출도 닫히고 처리 자리를 돌려준다 (`test_closing_the_stream_early_closes_the_model_call`). BE 에 따로 취소 기능을 만들 필요는 없고, FE 의 중지 → BE 가 AI 로의 연결을 닫는 것이면 된다. 중지한 답을 대화 기록에 남길지는 BE 가 정한다 |
| 제한 시간 | AI 쪽 전체 115초 — 첫 글자를 기다리는 시간과 스트림이 이 마감 하나를 나눠 쓴다 (BE 호출 제한 130초 안쪽). 넘으면 첫 글자 전에는 504 `MODEL_TIMEOUT`, 글이 나온 뒤에는 `error` 이벤트 (`retryable: true`). 글자가 한동안 안 나오는 것과는 다르다 — 모델이 생각하는 동안에는 `thinking` 이벤트가 온다 |
| 실패 화면을 나눌 기준 | `retryable` 로 나눈다. **true** (`AI_BUSY` · `SERVICE_UNAVAILABLE` · `MODEL_TIMEOUT` · `MODEL_PROVIDER_ERROR`) 는 "다시 시도" 버튼, **false** (`INVALID_MODEL_OUTPUT` 모델이 거절, `INPUT_TOO_LARGE` 대화가 너무 길다 → BE 가 오래된 턴을 줄여야 한다, `VALIDATION_FAILED`) 는 같은 요청을 다시 보내도 같으므로 버튼을 두지 않는다 |
| 오류 문구 | `message` 는 계약의 고정 문구라 로그·진단용이다 (예: "모델 처리 시간이 초과되었습니다."). 사용자에게 보일 문구는 `code` 로 FE 가 정한다 |
| 첫 글자 전 실패 vs 중간 실패 | 첫 글자 전: HTTP 오류 (JSON). 중간: `error` 이벤트 뒤 스트림 종료. 그러므로 `200` + `text/event-stream` 이면 말풍선을 만들고, 그 안에서 `error` 가 오면 그 말풍선을 "실패" 로 바꾸면 된다 (받은 글은 남아 있다) |

## BE 연동 메모

1. BE 는 사용자가 "이해가 안 됨" 을 누르면 카드(근거 A·B, 외부 지식, 결론)로 `knowledge` 를 만들고 대화 기록을 저장한다. 다음 턴마다 기록 전체를 `messages` 로 보낸다.
2. FE 에는 BE 가 SSE 를 그대로 흘려 보내면 된다 (`Cache-Control: no-cache`, 프록시 버퍼링 끄기 `X-Accel-Buffering: no`). FE 는 `delta` 를 이어 붙여 그리면 된다.
3. 답이 다 오면 (`done`) BE 가 그 답을 `assistant` 메시지로 기록 저장한다. 중간에 끊겼다면 받은 만큼만 저장할지는 BE 가 정한다.
4. 챗봇의 `knowledge.external` 의 url 은 새 지식 만들기 응답의 `external` (검색 결과에서 확인된 URL) 을 그대로 쓴다.

## 설정

| 환경 변수 | 뜻 |
| --- | --- |
| `AI_CHAT_LLM` | `1` 이면 챗봇을 켠다 (**기본 꺼짐**: 배포에서는 release manifest 의 `flags.chat` 으로 명시적으로 켠다, [deploy.md](deploy.md). 로컬 playground 는 켠다) |
| `AI_CHAT_LLM_PROVIDER` · `_MODEL` · `_EFFORT` | 없으면 `AI_LLM_*` → `ANTHROPIC_API_KEY` 의 anthropic. **모델은 기본 `claude-sonnet-5-5`, effort `low`** (쉽게 풀어 쓰는 말하기라 큰 모델이 필요 없고 첫 글자가 빨리 나온다). 스트리밍은 anthropic 만 지원한다 |

## 실제 측정 (이 머신, Sonnet 5.5, 3턴 대화, 한 번씩)

| 턴 | 첫 글자 | 끝까지 | 조각 수 |
| --- | --- | --- | --- |
| 1. "이중 버퍼링이 뭐예요?" | 2.3초 | 5.1초 | 83 |
| 2. "이해가 안 돼요" | 1.8초 | 4.8초 | 90 |
| 3. "그럼 O_DIRECT 는 왜 쓰는 거예요?" | 3.3초 | 5.6초 | 72 |

- 2번째 턴은 앞의 설명을 되풀이하지 않고 도서관 비유로 바꿔 말했고, 3번째 턴은 그 비유를 이어 쓰면서 카드에 없는 내용은 "카드에는 없지만" 이라고 밝혔다.
- system 프롬프트(카드 포함)는 캐시되어 2턴부터 입력 토큰 약 1,140개가 캐시에서 읽혔다.

## playground

`uv run python run_playground.py` → 지식 제안 탭에서 새 지식 카드의 **이해가 안 됨** 을 누르면 이 API 로 대화가 시작된다 (첫 말 "이해가 안 돼요" 를 자동으로 보내고, 입력칸에서 이어서 물을 수 있다).
답이 토큰마다 흘러오고, `중단` 으로 멈출 수 있다. **챗봇 탭** 에서는 카드를 골라(샘플 · 지식 제안에서 만든 카드 · 직접 고친 JSON) 바로 대화할 수 있고, 말풍선 아래에 첫 글자까지 걸린 시간과 조각 수가 실시간으로 보인다. 화면은 실제 `/internal/v1/chat` 을 서비스 토큰과 본문 해시로 호출한다 (`/dev/*` 가 아니다).
