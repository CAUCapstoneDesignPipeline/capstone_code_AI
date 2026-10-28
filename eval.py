"""평가 한 번에: `uv run python eval.py` (= `uv run ai eval`).

  ① 개념 추출 재현율·정밀도 (dev / test)   ④ 정답 노트 쌍 Recall@10·MRR   ④ 숨겼다 다시 찾기   ⑤ 설명 사람 판정 일치율
값마다 95% 신뢰 구간을 붙이고, data/private/eval/ 의 지난 보고서와 비교한다. 옵션은 `uv run python eval.py --help`.
"""

import sys

from app.cli import eval_main

if __name__ == "__main__":
    eval_main(sys.argv[1:])
