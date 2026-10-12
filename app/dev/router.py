"""개발용 playground (AI_DEV_UI=1 또는 `ai serve --dev` 일 때만 붙는다).

  GET /dev/              playground 화면
  GET /dev/config        화면 초기값 (profile 목록, 팀 관계 유형 초안, 로컬 접속이면 서비스 토큰)
  GET /dev/notes         data/private/notes 의 노트 id 목록
  GET /dev/notes/{id}    노트 원문 (NFC)
  GET/POST/DELETE /dev/llm   관계 판정 LLM (extract-hybrid-1) 을 실행 중에 고르거나 끈다.
                             API 키는 서버 메모리에만 두고, 저장·로그·응답에 넣지 않는다. 로컬 접속만 받는다

배포용이 아니다. 로컬 노트를 읽어 보여 주므로 127.0.0.1 에서만 띄운다.
"""

import json
import unicodedata
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.config import API_PREFIX, EMBEDDING_PROFILES, EXTRACT_PROFILE, EXTRACT_PROFILES
from app.services.llm import DEFAULT_MODELS, PROVIDERS, LLMConfig, make_llm
from app.services.relations import RelationTyper

router = APIRouter(prefix="/dev", include_in_schema=False)

NOTES_DIR = Path("data/private/notes")
RELATION_TYPES = Path("experiments/extract_v0/relation_types.json")
PAGE = Path(__file__).with_name("playground.html")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _note_paths() -> dict[str, Path]:
    return {_nfc(p.stem): p for p in sorted(NOTES_DIR.glob("*")) if p.suffix in (".md", ".txt")}


@router.get("/", response_class=HTMLResponse)
async def page():
    return PAGE.read_text(encoding="utf-8")


@router.get("/config")
async def config(request: Request):
    types = []
    if RELATION_TYPES.exists():
        types = [{"code": t["name"], "label": t["name"], "description": t["description"]}
                 for t in json.loads(RELATION_TYPES.read_text(encoding="utf-8"))["types"]]
    loopback = _loopback(request)
    st = request.app.state
    return {
        "llm": _llm_status(st),
        "llmProviders": [{"id": p, "label": LLM_LABELS[p], "defaultModel": DEFAULT_MODELS.get(p, ""),
                          "keyEnv": KEY_ENV[p]} for p in PROVIDERS],
        "apiPrefix": API_PREFIX,
        # 로컬 접속일 때만 토큰을 채워 준다 (직접 붙여넣지 않아도 되게)
        "token": st.settings.service_token if loopback else None,
        "extractProfiles": [{"id": p.id, "relations": p.relations, "default": p.id == EXTRACT_PROFILE}
                            for p in EXTRACT_PROFILES.values()],
        "embeddingProfiles": [{"id": p.id, "dims": p.dims, "loaded": p.id in st.embedders}
                              for p in EMBEDDING_PROFILES.values()],
        "relationTypes": types,
    }


@router.get("/notes")
async def notes():
    return sorted(_note_paths())


@router.get("/notes/{note_id:path}")
async def note(note_id: str):
    path = _note_paths().get(_nfc(note_id))
    if path is None:
        raise HTTPException(404)
    return {"id": _nfc(note_id), "body": _nfc(path.read_text(encoding="utf-8"))}


# ---- 관계 판정 LLM 고르기 ----

LLM_LABELS = {"anthropic": "Anthropic · Claude", "openai": "OpenAI", "google": "Google · Gemini"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY"}


def _loopback(request: Request) -> bool:
    return bool(request.client and request.client.host in ("127.0.0.1", "::1"))


def _llm_status(state) -> dict | None:
    t = state.typer
    return {"provider": t.provider, "model": t.model} if t is not None else None


class LlmChoice(BaseModel):
    provider: str
    model: str = ""
    apiKey: str = ""        # 비우면 서버 환경 변수 (ANTHROPIC_API_KEY 등) 를 쓴다


@router.get("/llm")
async def get_llm(request: Request):
    return {"llm": _llm_status(request.app.state)}


@router.post("/llm")
async def set_llm(choice: LlmChoice, request: Request):
    if not _loopback(request):
        raise HTTPException(403, "로컬에서만 바꿀 수 있다")
    provider = choice.provider.strip().lower()
    if provider not in PROVIDERS:
        raise HTTPException(400, f"제공자는 {', '.join(PROVIDERS)} 중 하나")
    model = choice.model.strip() or DEFAULT_MODELS.get(provider, "")
    if not model:
        raise HTTPException(400, f"{LLM_LABELS[provider]} 은 모델 id 를 직접 적어야 한다")
    cfg = LLMConfig(provider, model, api_key=choice.apiKey.strip() or None)
    try:   # client 를 만들 때 키가 없으면 SDK 가 바로 오류를 낸다 (openai, google)
        llm = await run_in_threadpool(make_llm, cfg)
    except Exception as e:
        raise HTTPException(400, f"{LLM_LABELS[provider]} client 를 만들지 못했다: {type(e).__name__}. "
                                 f"API 키 또는 {KEY_ENV[provider]} 를 확인한다")
    st = request.app.state
    st.typer = RelationTyper(llm, st.settings.relation_cache)
    return {"llm": _llm_status(st)}


@router.delete("/llm")
async def clear_llm(request: Request):
    if not _loopback(request):
        raise HTTPException(403, "로컬에서만 바꿀 수 있다")
    request.app.state.typer = None
    return {"llm": None}
