# 합성 노트 (조건부 결론 평가용)

사실 명세 → 다른 모델이 노트로 씀 → 검사 → 기대 결과와 비교. 노트는 합성이고 양이 많아 `data/private/synthetic/` (git 무시)에 둔다.

```bash
uv run python experiments/synthetic/build.py --notes 1000                 # 노트·정답 만들기 (Sonnet 이 쓴다)
uv run python experiments/synthetic/build.py --pilot 40 --out data/private/synthetic-pilot   # 작게 먼저
uv run python experiments/synthetic/evaluate.py [--limit 100] [--sound]   # 추출·도출 평가, 문체별 표
```

- **명세** (`catalog.py`, `build.py`): 알고리즘 20개 × 구성 요소 23개 템플릿 (교과서에 맞는 횟수·비용). 장면은 결론 42% · 보류 20% ·
  무관 15% · 이미 앎 10% · 기각 8% · 모순 5%. 변수 이름·연산 이름·언어(ko, en, ko-en)·형식(글머리표, 표, 산문, LaTeX, 섞음)·잡음을 사례마다 바꾼다.
- **기대 결과**는 명세의 사실에 규칙 엔진을 돌려 얻는다. 그래서 이 데이터가 재는 것은 규칙이 아니라 **문체가 달라도 사실을 제대로 뽑는가**다.
  규칙이 맞는지는 `docs/team/data` 의 사람이 쓴 사례로 본다.
- **검사**: 사실마다 quote 가 본문에 글자 그대로 있어야 하고, 명세에 없는 Big-O 식이 없어야 하며, 숨기기로 한 사실(보류 장면의 빠진 횟수·비용·크기)을
  노트가 말하면 (다른 호출로 판정) 다시 쓰게 한다. 두 번 안 되면 그 사례를 버린다.
- **한계**: 쓰는 모델이 하나라 문체 폭이 실제 학생 노트보다 좁다 (잡음이 깔끔하다). 알고리즘·구성 요소가 템플릿 안이다.
  기대 결과가 AI 가 쓴 것이라 사람이 확인한 정답이 아니다. 실제 노트의 성능은 `experiments/derivation/real_notes.py` 로 따로 본다.
