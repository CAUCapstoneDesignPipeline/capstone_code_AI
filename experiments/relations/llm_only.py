"""LLM 만으로 관계를 뽑는 기준선: 노트 전체와 개념 목록을 주고 관계(source, target, type)를 받는다. 후보 쌍·규칙은 쓰지 않는다.

정답을 만든 초안 프롬프트와 같은 기준(노트가 그렇게 말하는가, 구조가 드러내는 관계 포함, 유형 8개)이고 근거 문장은 요구하지 않는다 (출력이 짧아 빠르다 —
서비스에서는 근거를 코드가 원문에서 찾는다). 결과·지연은 data/private/cache/relations-llm-only 에 캐시한다 (같은 입력·모델·프롬프트면 다시 묻지 않는다).
정답 초안을 쓴 모델과 같은 계열(Claude)은 정답과 판단이 닮아 유리하다. 독립적인 읽기로 OpenAI·Google 모델도 같이 잰다.
"""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/private/cache/relations-llm-only"
PROMPT_VERSION = "relations-llm-only-1"
TYPES = ["is_a", "part_of", "causes", "requires", "used_for", "contrasts_with", "example_of", "analogous_to"]

SYSTEM = """너는 공부 노트를 읽고 개념 사이 관계를 뽑는다. 아래 개념 목록의 개념들 사이에서 노트가 말하는 관계를 빠짐없이, 그러나 노트가 말하지 않는 것은 넣지 않고 적는다.
판정 기준은 "노트가 그렇게 말하는가"다. 상식이나 노트 밖 지식으로 관계를 만들지 않는다. 두 개념이 같이 나오거나 나열되기만 한 것은 관계가 아니다.
- source·target 은 개념 목록의 표기 그대로 쓴다. 목록에 없는 것은 쓰지 않는다.
- type 은 관계 유형 중 하나다. 한 쌍에 서로 다른 유형이 모두 성립하면 각각 적는다. 방향이 있는 유형은 주어가 source, 목적어가 target 이다.
  contrasts_with·analogous_to 는 방향이 없다 (순서는 아무거나).
- 한 문장 안에서 서술된 관계뿐 아니라 제목·목록·표 구조가 드러내는 관계도 적는다
  (예: "## TCP" 아래 목록의 항목들이 TCP 의 구성 요소, 비교 표의 두 열이 서로 대비).

관계 유형:
- is_a: 주어가 목적어의 한 종류다 (예: TCP는 전송 프로토콜이다)
- part_of: 주어가 목적어의 구성 요소다
- causes: 주어가 목적어를 일으키거나 결과로 낳는다
- requires: 주어가 성립하거나 동작하려면 목적어가 필요하다
- used_for: 주어가 목적어를 위해 쓰인다
- contrasts_with: 노트가 주어와 목적어를 대비하거나 비교한다
- example_of: 주어가 목적어의 구체적 사례다
- analogous_to: 두 개념이 비슷한 원리로 동작한다"""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["relations"], "properties": {"relations": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["source", "target", "type"],
    "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string", "enum": TYPES}}}}}}


def make(spec: str):
    """spec = provider:model[:effort] → LLM 어댑터."""
    from app.services.llm import LLMConfig, make_llm
    provider, model, *rest = spec.split(":")
    return make_llm(LLMConfig(provider=provider, model=model, timeout=180.0, max_output_tokens=16000, effort=(rest[0] if rest else "medium")))


def ask(llm, labels: list[str], text: str):
    """→ ([(source, target, type)], 초). 캐시에 있으면 처음 불렀을 때의 지연을 돌려준다."""
    from app.services.llm import LLMError
    user = "<개념 목록>\n" + "\n".join(f"- {c}" for c in labels) + "\n</개념 목록>\n\n<노트>\n" + text + "\n</노트>"
    key = hashlib.sha256(json.dumps([llm.provider, llm.model, getattr(llm, "effort", None), PROMPT_VERSION, SYSTEM, user], ensure_ascii=False).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
    else:
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                data = llm.complete_json(SYSTEM, user, SCHEMA, "relations").data
                rec = {"relations": data.get("relations", []), "seconds": time.perf_counter() - t0}
                break
            except LLMError as e:
                if e.kind == "refusal":
                    rec = {"relations": [], "seconds": time.perf_counter() - t0, "refused": True}
                    break
                if e.kind not in ("timeout", "provider", "rate_limit") or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return [(r["source"], r["target"], r["type"]) for r in rec["relations"]], rec["seconds"]
