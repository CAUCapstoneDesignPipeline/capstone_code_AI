"""규칙으로 정하지 못한 개념 쌍의 관계 유형을 LLM 으로 정한다 (선택). 제공자는 app.services.llm 에서 고른다.

- 입력은 노트 전체가 아니라 문장과 두 개념 표기뿐이다. 근거 위치는 이미 코드로 정해져 있다.
- 출력은 요청 policy 의 code 또는 none 으로 제한한다 (구조화 출력 enum).
- 결과는 hash(제공자, 모델, 프롬프트 버전, policy, 문장, 쌍) 로 디스크에 캐싱한다. 같은 입력은 다시 호출하지 않는다.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.services.llm import LLMError

PROMPT_VERSION = "relation-typing-1"
NONE = "none"
TyperError = LLMError        # 예전 이름 (kind: timeout | rate_limit | auth | provider | invalid_output)

SYSTEM = """너는 공부 노트의 한 문장을 읽고, 표시된 두 개념 사이의 관계를 판정한다.
판정 기준은 "그 문장에 그렇게 쓰여 있는가"다. 상식이나 노트 밖 지식으로 관계를 만들지 않는다.
관계 유형은 아래 목록의 code 중 하나만 고른다. 문장이 두 개념의 관계를 말하지 않으면 none 이다.
direction 은 a 가 주어이면 a_to_b, b 가 주어이면 b_to_a 다.
confidence 는 문장이 그 관계를 얼마나 분명히 말하는지 0~1 로 적는다.

관계 유형:
{types}"""


@dataclass(frozen=True)
class PairItem:
    item_id: str
    sentence: str
    a: str
    b: str


@dataclass(frozen=True)
class Typed:
    code: str           # policy code (none 은 결과에 넣지 않는다)
    a_is_source: bool
    confidence: float


class RelationTyper:
    def __init__(self, llm, cache_dir: Path | None, chunk_size: int = 40, max_items: int = 200):
        self.llm, self.cache_dir, self.chunk_size, self.max_items = llm, cache_dir, chunk_size, max_items
        self.prompt_version = PROMPT_VERSION

    @property
    def provider(self) -> str:
        return self.llm.provider

    @property
    def model(self) -> str:
        return self.llm.model

    def type_pairs(self, items: list[PairItem], policy_types: list[tuple[str, str, str]]) -> dict[str, Typed]:
        """item_id → Typed. none 이거나 판정하지 않은 쌍은 결과에 없다. 앞에서부터 max_items 개만 본다."""
        items = items[: self.max_items]
        policy_sig = json.dumps(sorted(policy_types), ensure_ascii=False)
        out: dict[str, Typed] = {}
        todo = []
        for it in items:
            cached = self._cache_get(self._key(policy_sig, it))
            if cached is None:
                todo.append(it)
            elif cached["code"] != NONE:
                out[it.item_id] = Typed(cached["code"], cached["a_is_source"], cached["confidence"])
        for i in range(0, len(todo), self.chunk_size):
            chunk = todo[i:i + self.chunk_size]
            for it, res in zip(chunk, self._call(chunk, policy_types)):
                self._cache_put(self._key(policy_sig, it), res)
                if res["code"] != NONE:
                    out[it.item_id] = Typed(res["code"], res["a_is_source"], res["confidence"])
        return out

    def _call(self, chunk: list[PairItem], policy_types) -> list[dict]:
        codes = [c for c, _, _ in policy_types]
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "typeCode": {"type": "string", "enum": codes + [NONE]},
                    "direction": {"type": "string", "enum": ["a_to_b", "b_to_a"]},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "typeCode", "direction", "confidence"],
                "additionalProperties": False,
            }}},
            "required": ["items"],
            "additionalProperties": False,
        }
        types = "\n".join(f"- {c}: {label} — {desc}" for c, label, desc in policy_types)
        payload = [{"id": it.item_id, "sentence": it.sentence, "a": it.a, "b": it.b} for it in chunk]
        result = self.llm.complete_json(SYSTEM.format(types=types), json.dumps(payload, ensure_ascii=False),
                                        schema, name="relation_types")
        try:
            rows = {r["id"]: r for r in result.data["items"]}
        except (KeyError, TypeError) as e:
            raise LLMError("invalid_output", str(e))

        results = []
        for it in chunk:
            r = rows.get(it.item_id)
            if r is None or r.get("typeCode") not in codes + [NONE]:
                results.append({"code": NONE, "a_is_source": True, "confidence": 0.0})   # 빠진 항목은 none
                continue
            conf = r.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else 0.0
            results.append({"code": r["typeCode"], "a_is_source": r.get("direction") == "a_to_b",
                            "confidence": round(min(1.0, max(0.0, conf)), 4)})
        return results

    def _key(self, policy_sig: str, it: PairItem) -> str:
        raw = json.dumps([self.llm.provider, self.llm.model, self.prompt_version, policy_sig, it.sentence, it.a, it.b],
                         ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str):
        if self.cache_dir:
            p = self.cache_dir / f"{key}.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, value: dict):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
