# AI 서버 배포 (Docker)

기존 AWS 구성은 [서버 아키텍처](../../capstone_docs/architecture/server-architecture.md)의 같은 EC2 host 별도 컨테이너 예시다. GCP 목표는 [GCP 계획](../../capstone_docs/development/gcp-infrastructure-plan.md)의 서울 private AI VM `e2-highmem-2`(2 vCPU·16 GiB)다. 두 환경의 네트워크·비밀 주입 방식을 혼용하지 않는다.

이 체크아웃에는 GCP 운영 Compose·host 설치·배포 workflow가 아직 없다. 아래 Docker/Compose 제한값은 운영 시작값을 명시한 예시이며 GCP 배포 완료가 아니다. 다른 작업 브랜치의 배포 구현·검증과도 구분한다.

## 이미지

```bash
docker build -t capstone-ai --build-arg SOURCE_REVISION=$(git rev-parse HEAD) .
```

- 의존성은 `uv.lock` 그대로 설치한다 (`uv sync --frozen --no-dev`). Python 3.11, non-root(uid 10001), 약 690 MB (대부분 Kiwi 형태소 모델).
- 이미지에 **넣지 않는 것**: `data/`(노트·IDF·캐시), `.env`·API 키, 개발용 playground(`app/dev`). `.dockerignore` 가 허용 목록이다.
- 임베딩 모델(e5)을 쓰려면 `--build-arg UV_EXTRAS="--extra embeddings"` (torch 포함, 이미지가 커진다). 기본은 넣지 않는다.
- 운영 이미지의 시작값은 uvicorn worker 1개·`AI_MAX_INFLIGHT=1`이다. 초과 요청은 429 `AI_BUSY`로 거부한다. 컨테이너 CPU 1.5개·메모리 10 GiB는 실행 옵션/Compose에서 지정하며 Dockerfile만으로 강제되지 않는다.
- 로컬 개발은 `.env.example`의 동시성4와 코드의 미설정 기본8을 유지한다. 운영에서는 개발 환경 파일을 그대로 전달하지 않는다.
- `HEALTHCHECK` 는 `/internal/v1/health/live` 를 서비스 토큰으로 부른다 (`deploy/healthcheck.py`).

## 실행할 때 주는 것

| 환경 변수 / 파일 | 필수 | 설명 |
| --- | --- | --- |
| `AI_SERVICE_TOKEN` | 예 | BE 서비스 secret (256-bit 이상 무작위). 운영에서는 SSM 의 `CAPSTONE_AI_SERVICE_TOKEN` 을 이 이름으로 넣는다. 없으면 모든 요청 503 |
| `ANTHROPIC_API_KEY` | 예 | 지식 제안(`explain-llm-8`)은 항상 LLM 이 쓴다. 없으면 `/explanations` 만 503, 나머지는 동작 |
| IDF 파일 → `/run/ai/idf.json` (읽기 전용 mount) | 예 | 개념 점수의 기준 통계. 없거나 코드와 맞지 않으면 추출 profile 이 준비되지 않아 readiness 503 |
| `/var/cache/ai` volume | 권장 | 추출·관계·LLM 설명 캐시. 같은 입력의 LLM 호출을 다시 과금하지 않는다 |
| `AI_EXPLAIN_LLM_MODEL` / `AI_EXPLAIN_LLM_EFFORT` | 아니오 | 기본 `claude-opus-5-5`, `medium` |
| `AI_RELATION_LLM=1` | 권장 | 관계를 LLM 으로 보탠다 (`extract-hybrid-2`, `links-hybrid-1`). 끄면 노트 사이 경로가 거의 없다. effort 기본 `low`. 계산용 사실 추출 `facts-llm-1` 도 이것으로 켜진다 (effort 기본 `medium`, `AI_FACTS_LLM_*`) |
| `AI_VERIFY_LLM` | 아니오 | 기본 1. LLM 간선 검증 `edges-llm-1`·`edges-llm-2`, LLM 설명 검증 `explanation-hybrid-2`, LLM 병합 판정 `merge-hybrid-1` (effort `low`), 조건부 결론 판정 `derive-sound-1` (effort `medium`, `AI_SOUND_LLM_*`) |
| `AI_EMBEDDING_PROFILES` | 아니오 | 켤 임베딩 profile. 별칭 병합 후보 찾기는 `embeddings-bge-m3-1` (약 2 GB 메모리, [선택 근거](eval/alias-embeddings.md)). embeddings extra 로 빌드한 이미지만 |
| `AI_MAX_INFLIGHT`, `AI_LLM_TIMEOUT_SECONDS` | 아니오 | 동시성, LLM 한 번 호출 제한(기본 90초) |

`AI_DEV_UI` 는 이미지에서 `0` 이고 playground 코드도 없다. 켜면 시작하지 못한다 (의도된 동작).

## IDF 만들기

IDF 는 **이미지와 같은 commit 의 코드로** 만든다 (추출 설정·코드 hash 가 IDF 에 기록되고, 다르면 서버가 쓰지 않는다).
노트 원문에서 나온 낱말 통계이므로 이미지·저장소에 넣지 않고 host 에만 둔다.

```bash
uv run ai concepts build-idf --notes <참고 노트 폴더> --out /srv/capstone/ai/idf.json
```

## 단독 실행 (확인용)

```bash
docker run -d --name capstone-ai -p 127.0.0.1:8000:8000 \
  --cpus=1.5 --memory=10g -e AI_MAX_INFLIGHT=1 \
  -e AI_SERVICE_TOKEN -e ANTHROPIC_API_KEY \
  -v /srv/capstone/ai/idf.json:/run/ai/idf.json:ro -v capstone-ai-cache:/var/cache/ai \
  capstone-ai
```

`-e NAME` 처럼 값 없이 넘기면 셸 환경의 값이 들어가고 명령 기록에 남지 않는다. `.env` 전체를 `--env-file` 로 넘기지 않는다
(다른 서비스의 비밀값까지 컨테이너에 들어간다).

## Compose (기존 AWS host 예시, 실행하지 않음)

```yaml
services:
  ai:
    image: <ECR>/capstone-ai@sha256:<digest>      # digest 로 고정해 재현·rollback
    cpus: 1.5
    mem_limit: 10g
    environment:
      AI_MAX_INFLIGHT: "1"
      AI_DEV_UI: "0"
      AI_SERVICE_TOKEN: ${CAPSTONE_AI_SERVICE_TOKEN}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    volumes:
      - /srv/capstone/ai/idf.json:/run/ai/idf.json:ro
      - ai-cache:/var/cache/ai
    networks: [ai]                                # BE worker 만 붙는 network. ports 는 publish 하지 않는다
    restart: unless-stopped
    stop_grace_period: 130s                       # 진행 중 요청(최대 130초)을 마치고 내린다
networks:
  ai: {}                                          # 외부 LLM 호출이 나가야 하므로 internal: true 로 두지 않는다
volumes:
  ai-cache: {}
```

GCP 운영 Compose는 별도 구현·검토가 필요하다. AR digest, loopback8000과 private TLS8443, Secret Manager의 승인된 숫자 version, metadata 차단·health·rollback을 연결해야 한다. 실제 배포 전 1.5 CPU/10 GiB 제한 아래 모델 적재·동시2요청·429·timeout·OOM을 측정한다. 제품 AI는 BE 통합 및 별도 활성화 인수 전까지 off를 유지한다. 이 문서의 명령은 이번 로컬 설정 작업에서 실행하지 않았다.

배포 순서는 아키텍처 문서대로 **AI readiness 확인 → BE** 다. readiness 는 `GET /internal/v1/health/ready`
(Bearer + `X-Request-Id`)가 200 이고 `profileVersions` 에 BE manifest 의 profile 이 모두 있으면 된다. readiness 는 LLM 을
호출하지 않는다 (키가 설정됐는지만 본다).

## 확인한 것 (로컬 Docker 28, arm64)

- 빌드 성공, `uid=10001` 로 실행, `HEALTHCHECK` healthy
- readiness 200: 그 commit 의 profile 목록이 `profileVersions` 에 나온다 (LLM profile 은 LLM 설정이 있을 때만). IDF 를 안 주면 readiness 503 (liveness 는 healthy)
- `/dev/` 404, 틀린 토큰 401
- 참조 클라이언트(`app/client.py`)로 컨테이너에 전체 흐름(note-links → 검증 → LLM 설명 → 설명 검증)을 불러 candidate 제안 1건
