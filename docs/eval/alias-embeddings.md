# 별칭 병합 임베딩 모델 선택 (bge-m3 vs multilingual-e5)

결론: **`embeddings-bge-m3-1`** (BAAI/bge-m3, revision `5617a9f61b028005a4858fdac845db406aefb181`, 1024차원, 앞붙임 없음).
임베딩은 **병합 후보 쌍을 찾는 데만** 쓰고, 합칠지는 `/merge-decisions` 가 정한다. 어느 모델도 cosine 문턱값만으로는
설계 기준(잘못 합침 5% 이하)을 지키며 합칠 수 없었다.

재현: `uv run --extra embeddings python experiments/alias_embeddings/compare.py --include-draft`
(결과 파일은 노트 내용이 들어 있어 `data/private/gold/` 에만 둔다)

## 시험 자료

- 정답 개념 초안 (`data/private/gold/concepts.csv`, 노트 8개, **검토 전 draft**). 같은 줄의 `|` 표기는 같은 개념,
  노트가 달라도 표기가 겹치면 같은 개념으로 묶었다.
- 표기 300개 · 개념 236개 · 같은 개념 쌍 80개 · 전체 쌍 44,850개. 정답이 검토되면 숫자가 바뀔 수 있다.
- CPU 만 (Apple Silicon, torch 스레드 4). EC2 에서는 더 느릴 수 있다.

## 결과

| 모델 | 차원 | 같은 개념이 최근접 1 / 5 / 10 안에 | F1 최대 (문턱값, 정밀도, 재현율) | 메모리 (추론 후 증가분) | 묶음 추론 | 1개 추론 |
| --- | --- | --- | --- | --- | --- | --- |
| **bge-m3** | 1024 | **48% / 72% / 77%** | 0.80, 46%, 39% | 약 1.6 GB | 4.5 ms/표기 | 77 ms |
| multilingual-e5-large | 1024 | 41% / 69% / 77% | 0.92, 43%, 35% | 약 1.6 GB | 5.7 ms/표기 | 80 ms |
| multilingual-e5-small | 384 | 32% / 62% / 68% | 0.94, 48%, 25% | 약 0.5 GB | 0.9 ms/표기 | 12 ms |

- 입력은 표기만 쓴다. "표기: 처음 나온 문장" 은 세 모델 모두 더 나빴다 (최근접 정확도 5~9%p 하락).
- 메모리는 가중치(bge-m3 2.27 GB fp32)를 처음 쓸 때 올라온다. 프로세스 전체는 약 2.0 GB (Python·torch 포함).
- 적재 시간은 캐시가 있을 때 약 1초. 처음에는 모델 파일(약 2.3 GB)을 Hugging Face 에서 받는다.

## 왜 문턱값으로 합치지 않는가

가장 가까운데 **다른** 개념 (bge-m3): `read-write 충돌 ↔ write-write 충돌` 0.93, `대칭키 방식 ↔ 비대칭키 방식` 0.89,
`Non-repeatable Read ↔ Repeatable Read` 0.89, `내부 단편화 ↔ 외부 단편화` 0.85.
가장 먼데 **같은** 개념: `MDP ↔ Markov Decision Process` 0.37, `WAL ↔ Write Ahead Logging` 0.38, `Deadlock ↔ 교착 상태` 0.44.

반대말·짝 개념이 약어·번역보다 가깝다. 그래서 흐름은 이렇게 둔다.

1. BE: 개념 표기를 `/embeddings` (`embeddings-bge-m3-1`) 로 벡터로 만들고, 개념마다 cosine 최근접 몇 개(예: 5)를 병합 후보 쌍으로 고른다.
2. `/merge-decisions` (`merge-det-1`): 표기가 같거나 노트가 같은 것이라 적었을 때(괄호 병기, 두 표기뿐인 정의 목록 줄,
   `X, Y라고 한다`, 약어 ↔ 풀어 쓴 말이 함께 나옴)만
   같다고 하고, 문맥(함께 쓰는 명사구)을 본다. 표기가 다르면 다름, cosine ≥ 0.92 면 애매. `WAL ↔ Write Ahead Logging`
   은 DB-WAL 노트에 둘 다 나오므로 이 규칙으로 합쳐진다 (시연 3번).
3. 노트가 적어 두지 않은 번역(`Deadlock ↔ 교착 상태`)은 `merge-det-1` 로는 합치지 않는다. `merge-hybrid-1` 은 이런 쌍을
   LLM 이 두 표기와 근거 문장만 보고 판정한다 (확신 0.8 이상만 같음, 주제가 다른 병합도 LLM 확인을 거친다).

## 전체 흐름 시험 (/embeddings → /merge-decisions)

`uv run --extra embeddings python experiments/alias_embeddings/merge_pipeline.py --include-draft`
BE 처럼 표기마다 bge-m3 최근접 5개를 후보 쌍으로 묶어 `merge-det-1` 에 보냈다 (같은 정답 초안, 같은 개념 쌍 80개).

| | 후보에 든 같은 개념 쌍 | same (맞음 / 잘못 합침) | 전체 재현율 |
| --- | --- | --- | --- |
| 괄호 병기·약어 규칙 | 45 / 80 | 5 / 0 | 6% |
| + 정의 목록 한 줄(`- Atomicity: 원자성`)·이름 붙이기(`X, Y라고 한다`) | 45 / 80 | 10 / 0 | 12% |
| **`merge-hybrid-1`** (규칙이 정하지 못한 쌍을 LLM 이 판정) | 45 / 80 | **40 / 1** | **50%** |

`merge-hybrid-1` (`--profile merge-hybrid-1`, 검증 LLM `claude-opus-5-5` effort low): 후보에 든 같은 개념 쌍 45개 중 40개를
합쳤고 잘못 합침은 1건(`대칭키 ↔ 대칭키 암호화`, 2%)으로 기준 5% 이하다. `Mutex ↔ 뮤텍스`, `Transaction ↔ 트랜잭션`,
`Page Table ↔ 페이지 테이블` 같은 외래어·번역을 잇는다. 애매(uncertain) 9건은 합치지 않는다. 남은 손실은 후보에 들지 못한
35쌍이다 (아래 1). 아래는 규칙만 쓸 때(`merge-det-1`)의 분석이다.

규칙만 쓰면 잘못 합침은 0이지만 재현율이 낮다. 남은 손실은 두 가지다.
1. **후보에 들지 못함** (35쌍): 약어·번역은 임베딩이 멀다 (`WAL ↔ Write Ahead Logging` 0.38). 같은 노트에 둘 다 있고 규칙이
   잡을 수 있는 쌍도 최근접 5개 밖이면 판정까지 가지 못한다.
2. **노트가 같다고 적지 않은 외래어** (`Mutex ↔ 뮤텍스`, `Page Table ↔ 페이지 테이블`, `Transaction ↔ 트랜잭션`):
   규칙으로는 안전하게 정할 수 없다. 설계 ② 3 의 애매 구간 LLM 판정(`merge-hybrid-1`)으로 해결했다.

## BE 와 나눌 것 (EC2 배치)

- bge-m3 를 켜면 AI 컨테이너 메모리가 약 2 GB 더 필요하다 (모델 1.6 GB + 여유). 모자라면 `embeddings-e5-small-1`
  (0.5 GB) 로 시작하거나, 설계 10장대로 정규화 키 병합만 배포한다.
- 이미지는 `--build-arg UV_EXTRAS="--extra embeddings"` 로 만들고 `AI_EMBEDDING_PROFILES=embeddings-bge-m3-1` 로 켠다.
  모델 파일은 시작할 때 고정 revision 으로 받으므로 Hugging Face 캐시(`HF_HOME`)를 volume 으로 두면 다시 받지 않는다.
- 노트 저장 때 새 개념 수만큼만 부른다. 개념 100개면 묶음으로 약 0.5초 (이 기계 기준).
