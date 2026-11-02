"""조건부 결론 도출 (R1 구현 적용 · R2 비용 합성).

  facts        계산용 사실 (종류·주체·역할·연산·식·원문 근거)
  bigo         Big-O 상한 식의 대수 (곱·합·대입·지배 항 지우기)
  rules        R1·R2 규칙 → conclusion / held / rejected / known (결정적)
  extract_llm  노트 → 사실 (LLM, quote 가 원문에 그대로 있어야 한다)       → /facts
  sound_llm    규칙을 통과한 결론의 원문 충실성·soundness 판정 (LLM)      → /derivations (derive-sound-1)
  cases        대표 사례 (docs/team/data) 읽기·채점
"""
