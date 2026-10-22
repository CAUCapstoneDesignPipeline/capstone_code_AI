"""컨테이너 HEALTHCHECK: GET /internal/v1/health/live 가 200 이면 0, 아니면 1.

health 도 서비스 Bearer 와 X-Request-Id 가 필요하다 (ai-api.md Health operation). 토큰은 컨테이너 환경 변수에서 읽고
출력하지 않는다. 표준 라이브러리만 쓴다 (2초 timeout, 계약의 health 호출 제한).
readiness(/health/ready)는 IDF·LLM 설정에 따라 503 일 수 있으므로 배포 순서 판단은 BE·배포 스크립트가 따로 한다.
"""

import os
import sys
import urllib.request
import uuid

port = os.environ.get("PORT", "8000")
req = urllib.request.Request(f"http://127.0.0.1:{port}/internal/v1/health/live", headers={
    "Authorization": f"Bearer {os.environ.get('AI_SERVICE_TOKEN', '')}",
    "X-Request-Id": str(uuid.uuid4()),
    "Accept": "application/json",
})
try:
    with urllib.request.urlopen(req, timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
