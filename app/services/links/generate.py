"""새 지식 만들기: 키워드 A 의 요소 + 키워드 B 의 요소 + 인터넷 지식 → 결론 (회의록 10/30 "사용자가 모르는 지식 생성").

결론을 만드는 주체는 LLM 이고, 웹 검색으로 두 문장을 이어 주는 외부 자료를 찾는다. 모델 말을 그대로 믿지 않고 코드가 확인한다:
  - 외부 출처 URL 은 이 호출의 검색 결과에 실제로 나온 것만 남긴다 (모델이 지어낸 URL 은 버린다). 하나도 안 남으면 가설(held).
  - fromA · fromB 는 노트 문장에서 그대로 옮긴 부분이어야 한다 (결론이 두 노트에 근거를 둔다는 표시). 아니면 가설.
  - 결론이 A 나 B 와 거의 같으면 새 지식이 아니다 (known).
  - 모델이 새로 말할 것이 없다 · 노트와 웹이 다르다고 하면 가설.
통과하면 결과를 `generated` 개념(app/schemas/knowledge.py) 으로 만든다: 소스에 A·B 의 페이지를 넣고, 외부 자료는 external 에 둔다.
결과는 hash(제공자, 모델, 프롬프트 버전, 두 문장) 로 디스크에 캐싱한다.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import ValidationError

from app.schemas.common import Span
from app.schemas.knowledge import Concept, ExternalSource, PageSource
from app.services.llm import LLMError

from .check import PROBLEMS

PROMPT_VERSION = "newknowledge-llm-4"      # 4: 결론이 입력 문장을 A·B 라벨로 가리키지 않게 (3 과 결과가 같아 3 의 저장 결과는 그대로 쓴다: COMPATIBLE). 2: 검색 한 번 · url·quote 만 / 1: 검색 3번까지 · title·supports 도 모델이 씀. 3: 결과를 구조화 출력이 아니라 submit_knowledge 도구로 받아 결론이 쓰이는 대로 보인다
COMPATIBLE = ("newknowledge-llm-3",)       # 이 prompt 로 만든 저장 결과도 쓴다 (라벨 문제는 확인 단계가 잡는다). 그래야 저장해 둔 카드를 모두 다시 만들지 않는다
SUBMIT = "submit_knowledge"
MAX_EXTERNAL = 3
MAX_SEARCHES = 1                          # 검색 한 번이 약 9초다. 한 번에 9개쯤 받으므로 두 번째는 대개 같은 내용이다
SIMILAR = 0.85          # 결론이 노트 문장과 이만큼 같으면 새 지식이 아니다

SYSTEM = """너는 공부 노트에서 뽑은 두 문장(A, B)과 웹에서 찾은 자료를 합쳐, 사용자가 아직 모르는 새 지식 하나를 만든다.
새 지식 = A 와 B 어느 한쪽에도 쓰여 있지 않고, 둘을 합친 뒤 외부 자료로 이어야 비로소 말할 수 있는 결론이다.

순서: (1) A 와 B 를 읽고 어떤 결론이 가능한지 정한다. (2) 웹 검색으로 그 결론을 이어 주거나 확인해 주는 자료를 찾는다. 검색은 한 번만 할 수 있으니 검색어를 신중히 고른다.
(3) 아래 규칙대로 결과를 만들어 submit_knowledge 도구를 한 번 호출해 제출한다 (일반 글로 답하지 않는다).

규칙:
- conclusion 은 쉬운 말로 쓴다. 해요체로, "~는 ~에요" 처럼 친절하게, 2~4문장. 전문 용어는 처음 나올 때 풀어 쓴다.
- conclusion 은 혼자 읽혀야 한다. 입력 문장을 "A", "B", "A 문장", "위 문장" 처럼 라벨로 가리키지 말고 키워드 이름이나 내용으로 말한다 (읽는 사람은 이 라벨이 무엇인지 모를 수 있다).
- newKeyword 는 이 결론이 다루는 핵심 키워드 하나 (짧은 명사구).
- fromA, fromB 는 결론에 쓴 부분을 A, B 문장에서 한 글자도 바꾸지 않고 그대로 옮긴 것이다 (문장 일부만 옮겨도 된다).
- external 은 실제로 검색해서 본 페이지만 넣는다 (많아야 3개). url 은 검색 결과에 나온 그대로 쓰고, quote 는 그 페이지가 말한 내용을 한 문장으로 옮긴 것이다.
- 노트의 두 문장과 웹 자료가 서로 맞지 않으면 status 를 conflict 로 하고 reason 에 어디가 다른지 한 문장으로 적는다.
- 합쳐도 새로 말할 수 있는 것이 없으면 status 를 no_new_knowledge 로 한다. 노트에 이미 있는 말을 바꿔 쓰는 것은 새 지식이 아니다.
- 지어내지 않는다. 확실하지 않으면 no_new_knowledge 다. status 가 ok 가 아니면 conclusion·newKeyword 는 빈 문자열, external 은 빈 목록이어도 된다.
- status 가 ok 이면 reason 은 빈 문자열이다 (이유는 conclusion 에 이미 있다). 아니면 한 문장으로 쓴다.
입력은 JSON 이다: {"A": {keyword, note, text}, "B": {keyword, note, text}}."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "newKeyword", "conclusion", "fromA", "fromB", "external", "reason"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "no_new_knowledge", "conflict"]},
        "newKeyword": {"type": "string"}, "conclusion": {"type": "string"},
        "fromA": {"type": "string"}, "fromB": {"type": "string"},
        "external": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["url", "quote"],
            "properties": {k: {"type": "string"} for k in ("url", "quote")}}},
        "reason": {"type": "string"},
    },
}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\*\*|__|`", "", t)).strip()


def _url(u: str) -> str:
    return u.strip().rstrip("/")


def side_sources(*sides: dict) -> list[dict]:
    """요소(A·B)가 기댄 페이지 구간들, 같은 구간은 한 번만. 노트에서 뽑은 요소는 자기 구간 하나이고, '이해함' 으로 받아들인 새 지식(origin=generated)은
    그것이 기댄 구간들(`sources`)이다. 반환: [{noteId, note, start, end, quote}]."""
    out, seen = [], set()
    for x in sides:
        for r in (x.get("sources") or [x]) if x.get("origin") == "generated" else [x]:
            key = (r["noteId"], r["start"], r["end"])
            if key not in seen:
                seen.add(key)
                out.append({"noteId": r["noteId"], "note": r.get("note", ""), "start": r["start"], "end": r["end"], "quote": r["quote"]})
    return out


def merge_external(external: list[dict], *sides: dict) -> list[dict]:
    """새 외부 자료 뒤에 받아들인 새 지식이 기댄 외부 자료를 덧붙인다 (URL 이 같으면 한 번만, 개념의 상한 10개)."""
    out, seen = [], set()
    for e in [*external, *(e for x in sides if x.get("origin") == "generated" for e in x.get("external") or [])]:
        if e["url"] not in seen:
            seen.add(e["url"])
            out.append(e)
    return out[:10]


class KnowledgeGenerator:
    def __init__(self, llm, cache_dir: Path | None, tool: str | None = None, checker=None):
        """llm: AnthropicLLM 처럼 complete_web(system, user, schema, name, max_searches, tool) 가 있는 것.
        tool: 검색 도구. 기본은 기본판(web_search_20250305): 같은 쌍에서 동적 필터링판과 결과가 같았고 호출당 입력 토큰이 6분의 1이다 (3천 대 2만)."""
        self.llm, self.cache_dir, self.tool, self.checker = llm, cache_dir, tool, checker      # checker: check.KnowledgeChecker (없으면 확인 단계를 건너뛴다)
        self.prompt_version = PROMPT_VERSION

    @property
    def model(self) -> str:
        return self.llm.model

    def generate(self, a: dict, b: dict, concept_id: str, on_event=None) -> dict:
        """a, b: {keyword, note, noteId, text, start, end, quote}. 반환: {status: candidate|held|known, reasons, ...}.
        on_event(dict): 진행 상황 — cached · start · (모델이 알리는 thinking · search_start · search · search_results · writing) · validating.
        콜백이 예외를 던지면 모델 호출을 멈춘다."""
        emit = on_event or (lambda e: None)
        key, hit = self._cached(a, b)
        if hit is not None:
            emit({"type": "cached"})
            return {**self._dropped_if_unreadable(self._verify(hit, a, b, concept_id, emit), key), "cached": True}
        emit({"type": "start", "model": self.llm.model, "tool": self.tool or ""})
        user = json.dumps({"A": {k: a[k] for k in ("keyword", "note", "text")}, "B": {k: b[k] for k in ("keyword", "note", "text")}},
                          ensure_ascii=False)
        extra = {"on_event": on_event, "submit": SUBMIT} if on_event else {"submit": SUBMIT}   # 결과는 submit_knowledge 도구로 받는다 (쓰이는 대로 보이게)
        res = (self.llm.complete_web(SYSTEM, user, SCHEMA, "new_knowledge", MAX_SEARCHES, self.tool, **extra) if self.tool
               else self.llm.complete_web(SYSTEM, user, SCHEMA, "new_knowledge", MAX_SEARCHES, **extra))
        emit({"type": "validating", "sources": len(res.sources)})
        out = self._check(res, a, b, concept_id)
        self._cache_put(key, out)                                  # 저장하는 것은 확인 전의 결과다 (확인은 따로 캐시하고, 생성을 다시 돌리지 않고 확인만 바꿀 수 있게)
        return {**self._dropped_if_unreadable(self._verify(out, a, b, concept_id, emit), key), "cached": False}

    def _verify(self, out: dict, a: dict, b: dict, concept_id: str, emit) -> dict:
        """코드 검사를 통과한 결과(candidate)를 check.KnowledgeChecker 로 한 번 더 확인한다. 구조 문제(주제 이탈 등)가 있으면 보류, 무관한 출처는 뺀다.
        확인이 실패(LLM 오류)하면 결과를 막지 않고 check.error 만 남긴다 (확인은 선택 단계라서 새 지식 자체를 잃지 않는다)."""
        if self.checker is None or out["status"] != "candidate":
            return out
        emit({"type": "checking"})
        info = {"model": self.checker.model, "promptVersion": self.checker.prompt_version}
        try:
            r = self.checker.check(a, b, out["newKeyword"], out["conclusion"], out["external"])
        except LLMError as e:
            return {**out, "check": {**info, "error": e.kind}}
        info.update(problems=r["problems"], warnings=r["warnings"], reason=r["reason"], dropped=r["dropped"], reach=r["reach"])
        if not r["ok"]:
            return {**out, "status": "held", "reasons": r["problems"] + ([r["reason"]] if r["reason"] else []), "concept": None, "check": info}
        if r["dropped"]:
            kept = self._concept(concept_id, a, b, out["conclusion"], r["external"])
            return {**out, "external": r["external"], "concept": kept, "check": info}
        return {**out, "check": info}

    @staticmethod
    def _check(res, a: dict, b: dict, concept_id: str) -> dict:
        d = res.data
        reasons: list[str] = []
        seen = {_url(s["url"]): s["url"] for s in res.sources}
        titles = {s["url"]: s.get("title", "") for s in res.sources}            # 제목은 모델이 쓰지 않고 검색 결과에서 가져온다
        external, dropped = [], 0
        for e in d.get("external", [])[:MAX_EXTERNAL]:
            real = seen.get(_url(e.get("url", "")))
            if real is None:
                dropped += 1
            elif all(real != x["url"] for x in external):
                external.append({"url": real, "title": titles.get(real, "").strip(), "quote": e.get("quote", "").strip()})
        conclusion, from_a, from_b = d.get("conclusion", "").strip(), d.get("fromA", "").strip(), d.get("fromB", "").strip()
        status = "candidate"
        if d.get("status") == "no_new_knowledge":
            status, reasons = "held", ["합쳐도 새로 말할 수 있는 것이 없다고 판단했다"]
        elif d.get("status") == "conflict":
            status, reasons = "held", ["노트의 문장과 웹 자료가 서로 맞지 않는다"]
        else:
            if len(conclusion) < 15:
                status, reasons = "held", ["결론이 비었거나 너무 짧다"]
            if not from_a or _norm(from_a) not in _norm(a["text"]):
                status = "held"
                reasons.append("A 문장에서 쓴 부분을 그대로 옮기지 못했다")
            if not from_b or _norm(from_b) not in _norm(b["text"]):
                status = "held"
                reasons.append("B 문장에서 쓴 부분을 그대로 옮기지 못했다")
            if not external:
                status = "held"
                reasons.append("검색에서 실제로 본 외부 출처가 없다" + (f" (지어낸 URL {dropped}개를 버렸다)" if dropped else ""))
            if status == "candidate" and any(SequenceMatcher(None, _norm(conclusion), _norm(x["text"])).ratio() >= SIMILAR for x in (a, b)):
                status, reasons = "known", ["결론이 노트 문장과 거의 같아 새 지식이 아니다"]
        concept = None
        if status == "candidate":
            try:
                concept = KnowledgeGenerator._concept(concept_id, a, b, conclusion, external)
            except (ValidationError, ValueError) as err:
                status, reasons = "held", [f"개념 형식을 만들지 못했다: {err.errors()[0]['msg'] if isinstance(err, ValidationError) else err}"]
        return {"status": status, "reasons": reasons, "newKeyword": d.get("newKeyword", "").strip(), "conclusion": conclusion,
                "fromA": from_a, "fromB": from_b, "external": external, "reason": d.get("reason", "").strip(), "concept": concept,
                "searches": res.searches, "model": res.model, "promptVersion": PROMPT_VERSION}

    @staticmethod
    def _concept(concept_id: str, a: dict, b: dict, conclusion: str, external: list[dict]) -> dict:
        """개념 형식으로. 한쪽이 '이해함' 으로 받아들인 새 지식(generated)이면 그것이 기댄 페이지와 외부 자료도 이 개념의 소스가 된다
        (회의록: generated 의 소스에는 도출에 쓴 개념들의 페이지를 모두 넣는다)."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return Concept(
            conceptId=concept_id, origin="generated", explanation=conclusion[:2000],
            sources=[PageSource(noteId=x["noteId"], noteVersion=1, span=Span(startOffset=x["start"], endOffset=x["end"], quote=x["quote"]))
                     for x in side_sources(a, b)],
            external=[ExternalSource(url=e["url"], title=((e.get("title") or "")[:100] or None), retrievedAt=e.get("retrievedAt") or now,
                                     quote=(e.get("quote") or None))
                      for e in merge_external(external, a, b)]).model_dump(mode="json")

    def _key(self, a: dict, b: dict, version: str | None = None) -> str:
        raw = json.dumps([self.llm.provider, self.llm.model, version or self.prompt_version, a["keyword"], a["text"], b["keyword"], b["text"]],
                         ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cached(self, a: dict, b: dict) -> tuple[str, dict | None]:
        """(저장할 열쇠, 저장된 결과). 지금 prompt 로 만든 것이 없으면 COMPATIBLE 한 이전 prompt 의 것도 쓴다."""
        for version in (self.prompt_version, *COMPATIBLE):
            hit = self._cache_get(self._key(a, b, version))
            if hit is not None:
                return self._key(a, b, version), hit
        return self._key(a, b), None

    def _dropped_if_unreadable(self, res: dict, key: str) -> dict:
        """확인 단계가 '결론이 A·B 라벨로 가리켜 혼자 읽히지 않는다' 고 보류한 카드는 저장해 둔 생성 결과를 버린다: 같은 결과가 다시 나오지 않고 다시 만들 수 있다."""
        if PROBLEMS["selfContained"] in ((res.get("check") or {}).get("problems") or []) and self.cache_dir:
            (self.cache_dir / f"{key}.json").unlink(missing_ok=True)
        return res

    def _cache_get(self, key: str):
        if self.cache_dir and (self.cache_dir / f"{key}.json").exists():
            return json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, value: dict):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
