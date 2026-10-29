# AI 서버 배포 (Docker)

배포 구조는 `docs` 저장소의 [architecture/server-architecture.md](../../docs/architecture/server-architecture.md)를 따른다:
같은 EC2 host 의 별도 컨테이너, BE worker 만 `ai` Docker network 로 `8000` 에 접근, host port 는 publish 하지 않는다.

## 이미지

```bash
docker build -t capstone-ai --build-arg SOURCE_REVISION=$(git rev-parse HEAD) .
```

- 의존성은 `uv.lock` 그대로 설치한다 (`uv sync --frozen --no-dev`). Python 3.11, non-root(uid 10001), 약 690 MB (대부분 Kiwi 형태소 모델).
- 이미지에 **넣지 않는 것**: `data/`(노트·IDF·캐시), `.env`·API 키, 개발용 playground(`app/dev`). `.dockerignore` 가 허용 목록이다.
- 임베딩 모델(e5)을 쓰려면 `--build-arg UV_EXTRAS="--extra embeddings"` (torch 포함, 이미지가 커진다). 기본은 넣지 않는다.
- 프로세스 1개(uvicorn worker 1). 동시 요청 수는 `AI_MAX_INFLIGHT` (기본 8, 넘으면 429 `AI_BUSY`). 요청 대부분이 LLM 응답을 기다리는 시간이라 CPU 보다 크게 둔다.
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
  -e AI_SERVICE_TOKEN -e ANTHROPIC_API_KEY \
  -v /srv/capstone/ai/idf.json:/run/ai/idf.json:ro -v capstone-ai-cache:/var/cache/ai \
  capstone-ai
```

`-e NAME` 처럼 값 없이 넘기면 셸 환경의 값이 들어가고 명령 기록에 남지 않는다. `.env` 전체를 `--env-file` 로 넘기지 않는다
(다른 서비스의 비밀값까지 컨테이너에 들어간다).

## Compose (운영 host 예시)

```yaml
services:
  ai:
    image: <ECR>/capstone-ai@sha256:<digest>      # digest 로 고정해 재현·rollback
    environment:
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

배포 순서는 아키텍처 문서대로 **AI readiness 확인 → BE** 다. readiness 는 `GET /internal/v1/health/ready`
(Bearer + `X-Request-Id`)가 200 이고 `profileVersions` 에 BE manifest 의 profile 이 모두 있으면 된다. readiness 는 LLM 을
호출하지 않는다 (키가 설정됐는지만 본다).

## 확인한 것 (로컬 Docker 28, arm64)

- 빌드 성공, `uid=10001` 로 실행, `HEALTHCHECK` healthy
- readiness 200: 그 commit 의 profile 목록이 `profileVersions` 에 나온다 (LLM profile 은 LLM 설정이 있을 때만). IDF 를 안 주면 readiness 503 (liveness 는 healthy)
- `/dev/` 404, 틀린 토큰 401
- 참조 클라이언트(`app/client.py`)로 컨테이너에 전체 흐름(note-links → 검증 → LLM 설명 → 설명 검증)을 불러 candidate 제안 1건

## GCP 이미지 발행 준비 (#6)

위 EC2/SSM/ECR 절차와 arm64 검증은 과거 배포 기록이다. 신규 GCP 발행은
`.github/workflows/gcp-publish-image.yml`과 `deploy/gcp/publish_preflight.py`를 사용한다.
로컬 코드 준비와 실제 AR 발행·GCP 인수는 별개이며 현재 원격 변경 승인은 없다.

- Python 3.11.18 / uv 0.12.19 공식 image index digest를 2026-10-30 registry 조회로 고정했다.
  운영 build는 `linux/amd64`, 기본 `--extra embeddings`, uv frozen, UID10001이다.
- BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`은 기존 profile을 그대로 쓴다.
  `deploy/gcp/image_smoke.py`는 합성 2문서로 IDF를 만들고 CPU 모델 적재, 1024차원 벡터,
  인증 live/ready·profile·잘못된 토큰·`/dev` 비노출을 검사한다. 실제 provider를 호출하지 않는다.
- `develop`의 수동 실행만 허용한다. 기본은 실행 안 함이며 `GCP_IMAGE_PUBLISH_ENABLED=true`,
  보호된 `image-publish` environment, BE #36의 AI publisher WIF/AR 출력이 필요하다.
  repo/owner 숫자 ID, ref, workflow를 검사하며 IAM 측 조건과 실제 token 교환도 별도 인수한다.
- `GCP_PROJECT_ID`, `GCP_REGION`, `AR_REPOSITORY=capstone-ai`, `GCP_IMAGE_SERVICE_ACCOUNT`,
  `GCP_WIF_PROVIDER`는 승인된 Terraform output으로 설정한다. 장기 JSON key를 사용하지 않는다.
- run ID와 attempt를 image tag/artifact 이름에 포함해 immutable tag를 덮지 않는다.
  배포 입력은 `image.json`의 정확한 source SHA와 digest이며 tag는 배포에 쓰지 않는다.
- AI publisher가 앱 AR·VM·DB·Secret Manager에 접근할 수 없는지는 실제 IAM 부정 시험이 남는다.
  CI 통과나 preflight는 WIF 정책·provider 호출·모델 품질의 증거가 아니다.

검증 명령(저장소 루트):

```bash
uv sync --frozen --python 3.11
uv run --frozen pytest -q
docker build --platform linux/amd64 --build-arg UV_EXTRAS="--extra embeddings" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t capstone-ai:gcp-check .
docker run --rm --platform linux/amd64 --entrypoint python \
  -v "$PWD/deploy/gcp/image_smoke.py:/opt/image_smoke.py:ro" \
  capstone-ai:gcp-check /opt/image_smoke.py
```

CPU smoke는 BGE-M3 모델 다운로드·적재를 수행한다. local arm64 빌드나 hash embedder 테스트를
이 검사로 대체하지 않는다. 비밀값·IDF·개인 노트·캐시는 build context 허용 목록 밖에 둔다.

## Private GCP runtime 및 release (#7)

`deploy/gcp/install.sh`는 Ubuntu 24.04 amd64 전용 설치 코드다. 실제 실행에는 별도 VM 설치 승인이 필요하다.
`host.example.json`을 Terraform output으로 채우고 `validate_host()` 검사와 SHA256을 검토한다.
`GCP_CLI_PACKAGE_VERSION`(apt exact version), `OPS_AGENT_VERSION`을 승인 계획에 고정한다.
설치기는 Google 공식 key/installer bytes의 조회 시 SHA256을 검증하고 서명된 저장소를 사용한다.
이 작업은 프로젝트 생성·secret 값 등록·CA 발급·provider/model 선정·제품 AI 공개를 수행하지 않는다.

### 서로 다른 네 가지 고정 기준

1. **이미지 source SHA**: CI가 검증·발행한 AI 코드와 image digest. BE 코드는 별개다.
2. **설치 tool SHA**: `install.sh reviewed-host.json <40hex>`의 clean checkout commit.
   `/opt/capstone-ai/REVISION`과 manifest `bootstrapRevision`이 일치해야 한다.
3. **manifest 문서 SHA**: `GCP_RELEASE_MANIFEST_SHA`로 승인된 별도 commit을 checkout한다.
   tool commit 안에 그 commit 자신을 참조하는 manifest를 넣지 않는다.
4. **설정 revision**: `host.json`의 key 정렬·compact UTF-8 JSON SHA256 (`manifest.digest(host)`).
   host 주소·bucket·secret 자원·내부 SAN 변경은 설정과 새 manifest를 함께 검토한다.

manifest v1은 `deploy/gcp/manifest.py`로 검증한다. 필수 필드:

| 필드 | 고정 값/조건 |
| --- | --- |
| `schemaVersion`, `kind` | `1`, `ai` |
| `releaseId`, `sourceSha`, `image` | 제한된 release ID, 전체 commit, 서울 `capstone-ai/ai@sha256:…` |
| `bootstrapRevision`, `configRevision` | 설치 tool commit, host 설정 canonical SHA256 |
| `model` | `embeddings-bge-m3-1`, `BAAI/bge-m3`, 기존 고정 revision |
| `profiles` | 준비되어야 할 전체 profile; dev profile 금지 |
| `idf` | AI artifact bucket/object, 양의 숫자 generation, SHA256, 일치 source SHA |
| `provider` | `name`, relation/facts/explain/verify/sound별 명시 `models`·`efforts`, timeout≤90초 |
| `flags` | relation/explain/verify 명시 boolean. 켜진 profile은 readiness와 대조 |
| `secrets` | serviceToken/providerKey/tlsBundle 전체 resource와 숫자 version |
| `tls` | zonal 내부 DNS SAN, public CA PEM SHA256 |
| `cache` | `responses: disabled`, `modelScope: pinned-revision` |

승인된 IDF/provider/CA가 없으면 실제 manifest를 만들거나 허위 준비 완료로 표시하지 않는다.
`host.example.json`의 `REPLACE_…`는 실행 검증에서 거부된다. IDF가 배포 source/config와 맞는지 실제 extractor의
readiness도 확인한다. TLS secret JSON은 `certificate`(leaf PEM), `privateKey`(leaf key), `caCertificate`(public CA)
세 필드만 받는다. CA 개인키는 이 JSON·VM·이미지·state에 넣지 않는다. 서명 담당자·갱신 책임은 별도 결정한다.
leaf SAN/체인/키 일치 및 잔여 유효기간 14일을 preflight에서 검사한다.

### 배포·취소·재부팅·복구

- AI 전용 release bucket의 `releases/<releaseId>.json`, `operations/<operationId>.json`은 generation 0
  create-only. 같은 내용 재실행은 허용하고 같은 ID의 다른 내용은 거부한다.
- host는 `flock`으로 직렬화하고, 변경 전에 fsync한 operation journal을 남긴다.
  GCS `state.json`의 `{current,previous}`를 읽은 generation으로만 교체한다. stale expected release는 기동 전에 거부한다.
- root 전용 release directory에 IDF·숫자 version secret·leaf TLS·Compose 설정을 준비하고 검증한다.
  현재 release는 준비 실패에 영향받지 않는다. config·model·IDF·secret·TLS가 하나의 복원 단위다.
- Nginx와 runtime의 새 요청을 drain한 뒤 기존 프로세스를 최대 180초 동안 종료한다.
  private IP:8443만 Nginx가 받고 Docker는 `127.0.0.1:8000`에만 publish한다.
  AI 내부 처리 115초, provider≤90초, BE 예산130초, Nginx160초, uvicorn graceful150초, Docker stop180초다.
  keep-alive160초를 처리 제한으로 사용하지 않는다.
- 새 컨테이너의 인증 ready/profile·TLS SAN·잘못된 토큰401 검증 후 CAS한다.
  CAS 응답 유실은 cloud state를 다시 읽어 실제 결과를 확인한다. 해석 불가 상태는 maintenance를 유지한다.
- 실패 시 알려진 이전 manifest 전체를 복원한다. 최초 배포 실패는 중지+maintenance다.
  GCP AI 배포 실패가 앱 `aiEnabled=false`나 CRUD를 바꾸지 않는다.
- IAP는 detached systemd operation만 제출한다. Actions 취소가 host 작업을 죽이지 않는다.
  재실행은 같은 operation ID와 같은 request를 사용하고 journal을 확인한다. runner 성공은 durable phase `succeeded`일 때만 기록한다.
- 재부팅 및 Docker 재시작은 `capstone-ai.service`가 미완료 journal·GCS state를 먼저 reconcile한다.
  컨테이너 restart policy는 `no`라서 검증 전 자동 재기동하지 않는다. SQL/DB rollback은 존재하지 않는다.
- rollback 요청은 state의 `previous`만 허용한다. installed tool/config revision이 달라 호환 검증에 실패하면
  tool/config 복구도 포함한 별도 계획이 필요하며, 예전 이미지만 무조건 기동하지 않는다.

### 권한·캐시·관측

`capstone-ai-metadata.service`는 Docker FORWARD에서 IPv4/IPv6 metadata를 차단한다. host의 ADC는 INPUT/OUTPUT
경로를 유지한다. 실제 VM에서 container→metadata 거부, host→ADC 정상, AI→SQL/앱 secret 거부를 각각 시험해야 한다.
실제 환경 검증 없이 이 규칙 존재만으로 격리를 완료했다고 보고하지 않는다.

현재 AI 결과 캐시 키에는 확정된 사용자 scope/삭제/TTL 계약이 없다. 운영 entrypoint `app.gcp_runtime`는
기존 알고리즘 변경 없이 원문·관계·설명·검증 응답 캐시를 비활성화한다. public 모델 파일만 고정 revision으로
별도 mount에 보존한다. 모델 cache의 용량·현행/이전 revision 보존 및 삭제는 운영 인수 시 정한다.
로그는 SDK 예외·원문·credential을 출력하지 않도록 runtime diagnostic만 남기며 Nginx access log는 끈다.
Docker 로그는 10 MiB×3으로 회전하고 Ops Agent는 host metrics만 수집한다.

실제 인수에는 cold/warm RSS·latency, 동시2요청·429·timeout·drain, TLS 만료 알림·CA 갱신, model cache 용량,
앱 VM에서 인증 ready/profile 및 잘못된 CA/token, 외부 8000/8443 차단, AI→SQL 거부, 재부팅/rollback,
Ops Agent 수신/알림을 포함한다. readiness는 provider를 호출하지 않는다. 실제 provider smoke는 승인된
합성 fixture·호출 수/비용 한도 안에서 별도 수행하고 비용 미측정을 0으로 보고하지 않는다.

### 원격 게시 및 실행 전 필요한 설정

- `image-publish`와 `ai-production` environment의 reviewer 및 develop deployment 정책.
- 숫자 owner/repo ID, develop, environment, exact caller `gcp-deploy.yml`, SHA로 고정한 reusable claim을
  대조한 AI WIF. publisher와 deployer는 다른 SA이며 앱 release bucket/앱 secret/SQL 권한이 없다.
- `GCP_AI_DEPLOY_TOOL_SHA`, `GCP_RELEASE_MANIFEST_SHA`, AI 전용 `GCP_VM_NAME`·`RELEASE_BUCKET`,
  project/zone/WIF/deployer SA. `GCP_DEPLOY_ENABLED` 기본 비활성. 이 설정 및 workflow 실행은 별도 승인 대상이다.
- 로컬-only commit은 Actions checkout할 수 없다. 원격 게시 후 ref 접근을 검증하고 WIF를 활성화한다.
- 이 배포는 private AI 서버 인수다. BE 기능 계약 승인이나 운영 사용자 대상 AI 공개를 대신하지 않는다.

AI caller는 실제 로컬 commit `40d9b6fe697f1ff5d633b75a57e8455e3263c636`의 reusable/tool에 고정했다.
이 commit은 원격 게시 전에는 GitHub에서 사용할 수 없다. 설치 tool SHA 및 `GCP_AI_DEPLOY_TOOL_SHA`,
WIF `job_workflow_ref`도 같은 값으로 대조한 후 사용한다. Manifest는 별도의 `GCP_RELEASE_MANIFEST_SHA`에서 읽는다.
