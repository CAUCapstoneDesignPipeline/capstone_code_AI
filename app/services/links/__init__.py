"""노트 사이 연결 (결정적). 설계 docs/concept-pipeline.md ③④ 의 기준 구현이다.

계약상 노트 사이 경로 탐색·순위는 BE 책임이다 (F-PATH-01, F-RANK-01). 여기 구현은 같은 방법을 AI 쪽에서
실험·평가하고 playground 로 확인하기 위한 것이다. BE 로 옮기거나 계약에 operation 을 더하는 것은 팀이 정한다.

  1. 노트마다 개념·관계 추출 (extract-det-3 와 같은 결정적 방법)
  2. 노트 사이의 같은 개념 묶기: 표기 정규화 + 노트 안의 확실한 별칭 (한·영 병기, 약어)
  3. 노트 쌍 점수: 공유 개념마다 양쪽 중요도의 곱 / log(1 + 그 개념이 나온 노트 수)  (Adamic-Adar)
     너무 흔한 허브 개념(전체 노트의 일정 비율 이상)은 빼다
  4. 연결 설명: 공유 개념 B 의 각 노트 근거 문장, 그리고 A–B (노트 1) · B–C (노트 2) 관계가 있으면 A–B–C 경로
"""

from .corpus import Corpus, build_corpus, build_entry, loose_key
from .linker import Bridge, Link, related_notes

__all__ = ["Bridge", "Corpus", "Link", "build_corpus", "build_entry", "loose_key", "related_notes"]
