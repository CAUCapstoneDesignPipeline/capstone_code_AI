# CAPSTONE AI 서버 (FastAPI, 내부 API /internal/v1). 배포: docs/deploy.md
#
#   docker build -t capstone-ai --build-arg SOURCE_REVISION=$(git rev-parse HEAD) .
#   docker run --rm -p 127.0.0.1:8000:8000 --env-file ai.env \
#     -v /srv/capstone/ai/idf.json:/run/ai/idf.json:ro -v capstone-ai-cache:/var/cache/ai capstone-ai
#
# 이미지에 넣지 않는 것: 노트·IDF(data/), .env·API 키, 개발용 playground(app/dev). 비밀값은 실행 시 환경 변수로만 준다.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

# ---- 빌드: uv 로 잠금 파일(uv.lock) 그대로 가상 환경을 만든다 ----
FROM ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 AS uv
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# 선택 기능: --build-arg UV_EXTRAS="--extra embeddings" 이면 sentence-transformers(임베딩 모델)를 넣는다
ARG UV_EXTRAS="--extra embeddings"
# 의존성 층을 먼저 만들어 코드만 바뀔 때 다시 받지 않게 한다
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${UV_EXTRAS}
COPY app ./app
# The dependency layer can retain an older local-project wheel when only app/ changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --reinstall-package ai ${UV_EXTRAS}

# ---- 실행 ----
FROM ${PYTHON_IMAGE}
ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/CAUCapstoneDesignPipeline/capstone_code_AI" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.title="capstone-ai"
RUN groupadd --gid 10001 capstone && useradd --uid 10001 --gid 10001 --no-create-home capstone \
    && mkdir -p /var/cache/ai /run/ai && chown 10001:10001 /var/cache/ai
COPY --from=build /opt/venv /opt/venv
COPY deploy/healthcheck.py /opt/healthcheck.py
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    AI_CONCEPT_IDF=/run/ai/idf.json \
    AI_CONCEPT_CACHE=/var/cache/ai/concepts \
    AI_RELATION_CACHE=/var/cache/ai/relations \
    AI_EXPLAIN_CACHE=/var/cache/ai/explanations \
    AI_VERIFY_CACHE=/var/cache/ai/verifications \
    HF_HOME=/var/cache/ai/huggingface \
    AI_DEV_UI=0
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=6 CMD ["python", "/opt/healthcheck.py"]
# 프로세스 1개: 형태소 분석기·임베딩 모델을 프로세스마다 올리지 않는다. 동시성은 AI_MAX_INFLIGHT 로 조절한다.
# keep-alive 는 BE 호출 제한(130s)보다 길게 둔다 (내부 Nginx read timeout 160s 와 맞춤)
CMD ["sh", "-c", "exec uvicorn app.gcp_runtime:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log --timeout-keep-alive 160 --timeout-graceful-shutdown 150"]
