"""테스트 폴더(tests/)를 import 경로에 올려 하위 폴더의 테스트가 tests/fake_decisions.py 같은 공용 도우미를 같은 이름으로 읽게 한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
