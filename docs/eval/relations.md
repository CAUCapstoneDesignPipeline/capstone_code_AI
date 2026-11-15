# 관계 추출 모델 비교 — 평가

한 노트 안의 개념 사이 관계(유형 8개 + 방향 + 근거)를 어떤 방법으로 뽑을지 같은 정답·같은 채점으로 견줬다. **개념은 정답 개념으로 고정**해 관계 추출만 잰다 (개념 추출의 오차가 섞이지 않게).
정답과 판정이 모두 AI 가 만든 것이라 사람이 검토하기 전의 수치다 ("읽을 때 주의" 참고).

## 정답과 채점

- **정답** (`experiments/relations/gold.py`): 정답 개념이 있는 노트 59개(과목 dev 23 · test 18, 논문 리뷰 18)에서 Opus 5.5 가 초안을 쓰고 Sonnet 5.5 가 검토했다 (받아들임 90~93%, 검토자가 더한 것 포함).
  관계 1,087개 (노트당 20~27개). 판정 기준은 "노트가 그렇게 말하는가"다 (상식으로 만들지 않는다). **약 60% 는 문장이 아니라 제목·목록·표 구조가 드러내는 관계**다 (예: "## Level 1: Read Committed" → Read Committed is_a Isolation Level).
- **정답이 불완전하다.** 정답에 없는 Sonnet 의 예측 관계를 Opus 가 노트에 비춰 판정하니 새 쌍의 69~78% 가 노트가 실제로 말하는 관계였다. 정답만으로 채점하면 정밀도가 42~46% 로 낮게 나오지만 어림한 실제 정밀도는 74~81% (유형까지 같음) · 85~88% (어떤 관계든) 였다.
- **풀링 판정** (`pool.py`, `pooled_eval.py`): 모든 방법의 예측 중 정답에 없는 것을 Opus 가 한 번씩 판정해 저장하고, 맞는 관계 집합 V = 정답 ∪ 판정자가 맞다고 본 것 (판정자가 고른 유형·방향)으로 모든 방법을 채점한다.
  strict = 쌍·유형·방향이 V 에 든 것, pair = 두 개념 사이 관계가 있다고 맞힌 것. V 는 방법을 더할 때마다 커지므로 방법은 같이 비교한다 (아래는 지금까지의 판정 전체 기준: V 는 dev 1,545 · test 1,607 · 논문 321).
- **후보 쌍** (`candidates.py`): 같은 문장 ∪ 같은 절(가장 안쪽 제목 아래) ∪ 제목에 나온 개념 ↔ 그 제목 절 안의 개념. 정답 쌍의 98% 를 담으면서 모든 쌍의 36~39% 만 남긴다 (노트당 과목 260~360쌍, 논문 65쌍).
  **기존 서비스의 후보 쌍(같은 문장·언급 3칸 안·사이 40자 이하)과 구조 규칙은 정답 쌍의 49~58% 에만 닿는다** — 문장 속 쌍만 보는 방법은 재현율 상한이 거기다.

## 결과 (strict F1 / pair F1, dev · test · 논문 리뷰)

| 방법 | 지연 (노트 하나) | strict F1 | pair F1 |
| --- | --- | --- | --- |
| 규칙 + 구조 (지금 서비스의 관계, LLM 없음) | 약 10 ms | .02 | .03~.04 |
| 구조·위치 특징만 학습 (Decisions 없음) | 수 ms | .18 / .16 / .11 | .56 / .57 / .52 |
| 규칙 + 구조 + LLM typer (Sonnet low, 서비스의 hybrid 와 같은 방식) | 중앙값 13~15 s (논문 4 s) | .24 / .27 / .40 | .35 / .38 / .53 |
| gpt-5.4-mini 만으로 (LLM-only) | 2~4 s | .32 / .28 / .40 | .60 / .56 / .66 |
| Decisions 객관식만 (쌍마다 질문 하나, 임계값 0.55) | 중앙값 0.8 s · p90 1.6 s | .35 / .35 / .31 | .58 / .60 / .58 |
| **구조 특징 + Decisions 확률을 학습한 분류기** (59노트 교차검증) | **약 1 s** | **.41 / .38 / .36** | **.66 / .66 / .62** |
| Haiku 5.5 만으로 | 7~21 s | .50 / .50 / .54 | .67 / .64 / .70 |
| **Sonnet 5.5 만으로** (노트 + 개념 목록 → 관계) | **3~7 s** | **.57 / .55 / .63** | **.72 / .70 / .78** |

- **LLM 이 노트 전체를 읽는 방법이 가장 낫다.** Sonnet 이 정답의 79% 를 찾고 낸 관계의 78~81% 가 맞다 (strict). 구조로 드러나는 관계를 잡는 것은 이 방법뿐이다.
- 지금 서비스의 규칙은 노트당 1개쯤만 내서 (정밀도 50~83%) 재현율이 1~2% 이고, LLM typer 를 붙여도 (후보 쌍이 문장 속 쌍뿐이라) strict F1 이 .24~.40 에 그치며 13초 넘게 걸린다.
- 빠른 쪽에서는 Decisions 확률이 유형 판단을 채운다: 구조 특징만으로는 쌍은 .56 정도 찾지만 유형·방향이 틀려 strict .16~.18, Decisions 를 더하면 .36~.41.
- Haiku 지연은 한 번에 4개씩 보냈을 때의 값이라 흔들린다 (Sonnet 보다 느린 것은 출력이 길어서일 수 있다).

## 읽을 때 주의

- **정답·판정이 AI 가 만든 것이다** (사람 검토 전). 판정자(Opus)가 정답 초안을 쓴 모델과 같고 Sonnet 이 검토에 참여했으므로 Claude 계열 방법이 정답과 판단이 닮아 유리하다. 독립적인 읽기로 gpt-5.4-mini 를 같이 쟀지만 작은 모델이라 낮은 점수가 그 탓인지 가려지지 않는다.
  강한 비-Claude 모델(gpt-5.5, Gemini Pro)은 지연이 10~40초라 이 비교에 넣지 않았다. 정답 몇십 개를 사람이 확인해야 한다.
- 풀링 판정은 풀에 낸 방법에 유리하다 (방법을 더하면 V 가 커져 다른 방법의 재현율이 내려간다). 순위는 풀이 커져도 바뀌지 않았지만 값은 바뀐다.
- 학습한 분류기는 59개 노트로 노트 단위 5겹 교차검증한 값이고 임계값을 dev 예측에서 골랐다. 교사(Sonnet 의 관계)가 정답이 아니므로 교사의 오류를 따라 한다. 학습 노트를 늘리면 나아질 수 있으나 재 보지 않았다.
- 개념은 정답 개념이다. 실제 서비스에서는 개념 추출의 오차가 더해진다. 응답에는 관계마다 근거 구간이 있어야 하는데 LLM-only 는 근거를 내지 않는다 (코드가 원문에서 두 개념이 함께 나온 문장·제목 줄을 찾아야 한다, 아직 구현하지 않았다).
- 유형 구분 자체가 흐리다 (같은 쌍에 used_for 와 requires 가 다 성립하는 식). pair 점수가 더 믿을 만하다.

## 재현

```bash
uv run --env-file .env python experiments/relations/gold.py                       # 정답 만들기 (Opus 초안 + Sonnet 검토, data/private/gold-relations)
uv run --env-file .env python experiments/relations/eval.py --methods reach rules   # 후보 쌍이 닿는 범위, 규칙 기준선
uv run --env-file .env python experiments/relations/adjudicate.py                  # 정답에 없는 예측의 표본 판정 (정답이 불완전한 정도)
uv run --env-file .env python experiments/relations/decisions_pairs.py             # Decisions 객관식 (쌍마다 질문 하나)
uv run --env-file .env python experiments/relations/pooled_eval.py --methods rules hybrid:anthropic:claude-sonnet-5-5:low llm:anthropic:claude-sonnet-5-5 llm:anthropic:claude-haiku-5-5 llm:openai:gpt-5.4-mini decisions:0.55
uv run --env-file .env python experiments/relations/pair_model.py [--no-decisions]  # 학습한 쌍 분류기 (교차검증)
```
