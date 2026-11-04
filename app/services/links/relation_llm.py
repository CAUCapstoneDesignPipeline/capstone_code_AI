"""두 키워드 노드가 실제로 관계가 있는지 노트 증거로 LLM 에 묻는다 (선택).

- 입력은 두 키워드의 설명(개념)과 연결 증거(같은 문장·제목과 본문·같은 문단, 또는 중간 키워드를 거친 간접 증거)뿐이다.
  판정 기준은 "그 증거에 두 키워드의 관계가 쓰여 있는가"다. 노트 밖 지식으로 관계를 만들지 않는다.
- 근거는 증거 id 로 인용하게 하고, 입력에 없는 id 는 버린다. 인용한 증거가 없으면 판정하지 않는다 (unsure).
  직접 증거(sentence·heading·paragraph)를 하나도 인용하지 못하면 related 를 weak 로 낮춘다.
- 결과는 hash(제공자, 모델, 프롬프트 버전, 입력) 로 디스크에 캐싱한다. 같은 입력은 다시 호출하지 않는다.
"""

import hashlib
import json
from pathlib import Path

from app.services.llm import LLMError

PROMPT_VERSION = "kwrel-llm-1"
NONE = "none"
TYPES = {
    "is_a": "한쪽이 다른 쪽의 한 종류다",
    "part_of": "한쪽이 다른 쪽의 구성 요소다",
    "causes": "한쪽이 다른 쪽을 일으키거나 결과로 낳는다",
    "requires": "한쪽이 성립하거나 동작하려면 다른 쪽이 필요하다",
    "used_for": "한쪽이 다른 쪽을 위해 쓰인다",
    "contrasts_with": "노트가 둘을 대비하거나 비교한다",
    "example_of": "한쪽이 다른 쪽의 구체적 사례다",
    "analogous_to": "두 개념이 비슷한 원리로 동작한다",
    "related": "위 유형으로 말하기 어렵지만 노트가 둘의 관계를 분명히 말한다",
}
VERDICTS = ("related", "weak", "unrelated")
DIRECT = ("sentence", "heading", "paragraph")

SYSTEM = """너는 공부 노트에서 뽑은 두 키워드(A, B)가 실제로 관계가 있는지 판정한다.
입력은 두 키워드의 설명과 증거 목록이다. 증거는 id(E1, E2, …), level, label, 노트 이름, 원문으로 되어 있다.
level 의미: sentence = 같은 문장에 둘 다 나옴 · heading = 한쪽이 제목이고 다른 쪽이 그 절 본문에 나옴 · paragraph = 같은 문단에 나옴 ·
path = 중간 키워드를 거친 간접 증거 (label 이 어느 두 키워드 사이의 증거인지 말해 준다).

판정 기준은 "증거에 두 키워드의 관계가 쓰여 있는가"다. 상식이나 노트 밖 지식으로 관계를 만들지 않는다.
같은 문장·문단에 나왔다는 사실만으로는 관계가 아니다. 한쪽이 다른 쪽을 설명·정의·비교·전제·사례로 다루는 내용이 있어야 한다.
같은 낱말이 다른 뜻으로 쓰였으면 관계가 없다.

verdict:
- related: 증거가 두 키워드의 관계를 직접 말한다.
- weak: 같은 맥락에서 함께 다루지만 관계 내용은 직접 말하지 않거나, 중간 키워드를 거친 간접 관계다.
- unrelated: 증거가 두 키워드의 관계를 말하지 않는다 (우연히 같이 나옴, 서로 다른 뜻).
type 은 아래 목록에서 하나만 고른다. related·weak 인데 어느 유형도 아니면 related 를 고르고, unrelated 면 none 이다.
direction: a_to_b 는 A 가 주어(예: A 는 B 의 한 종류), b_to_a 는 B 가 주어, 방향이 없거나 대칭이면 symmetric.
reason: 왜 그렇게 판정했는지 한국어 한두 문장. 증거에 쓰인 내용으로만 말한다.
evidence: 판정에 쓴 증거 id 들. related·weak 이면 하나 이상 반드시 인용한다.

관계 유형:
{types}"""


class KeywordRelationJudge:
    def __init__(self, llm, cache_dir: Path | None):
        self.llm, self.cache_dir = llm, cache_dir
        self.prompt_version = PROMPT_VERSION

    @property
    def provider(self) -> str:
        return self.llm.provider

    @property
    def model(self) -> str:
        return self.llm.model

    def judge(self, a: dict, b: dict, evidence: list[dict], on_event=None) -> dict:
        """a, b: {name, concepts: [설명]}. evidence: [{level, label?, note, text}] (id 는 여기서 E1.. 로 붙인다).
        반환: {verdict, type, direction, reason, evidence: [id], cached}. verdict 는 related | weak | unrelated | unsure.
        on_event(dict): cached · field(path, delta | value) — 모델이 쓰는 verdict·type·direction·reason 이 쓰이는 대로 (콜백이 예외를 던지면 호출을 멈춘다)."""
        items = [{"id": f"E{i}", "level": e["level"], "label": e.get("label", ""), "note": e["note"], "text": e["text"]}
                 for i, e in enumerate(evidence, 1)]
        payload = {"a": a, "b": b, "evidence": items}
        key = self._key(payload)
        cached = self._cache_get(key)
        if cached is not None:
            if on_event:
                on_event({"type": "cached"})
            return {**cached, "cached": True}
        result = self._call(payload, on_event)
        out = self._validate(result, items)
        self._cache_put(key, out)
        return {**out, "cached": False}

    def _call(self, payload: dict, on_event=None) -> dict:
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["verdict", "type", "direction", "reason", "evidence"],
            "properties": {
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "type": {"type": "string", "enum": [*TYPES, NONE]},
                "direction": {"type": "string", "enum": ["a_to_b", "b_to_a", "symmetric"]},
                "reason": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
        }
        system = SYSTEM.format(types="\n".join(f"- {c}: {d}" for c, d in TYPES.items()))
        user = json.dumps(payload, ensure_ascii=False)
        if on_event and hasattr(self.llm, "stream_json"):
            return self.llm.stream_json(system, user, schema, "keyword_relation", on_event=on_event).data
        return self.llm.complete_json(system, user, schema, name="keyword_relation").data

    @staticmethod
    def _validate(r: dict, items: list[dict]) -> dict:
        level = {i["id"]: i["level"] for i in items}
        try:
            verdict, typ, direction = r["verdict"], r["type"], r["direction"]
            reason, cited = str(r["reason"]).strip(), [x for x in r["evidence"] if x in level]
        except (KeyError, TypeError) as e:
            raise LLMError("invalid_output", str(e))
        if verdict not in VERDICTS or typ not in (*TYPES, NONE) or direction not in ("a_to_b", "b_to_a", "symmetric"):
            raise LLMError("invalid_output", "enum")
        cited = list(dict.fromkeys(cited))
        if verdict != "unrelated" and not cited:             # 근거를 인용하지 못한 판정은 받아들이지 않는다
            return {"verdict": "unsure", "type": NONE, "direction": "symmetric", "evidence": [],
                    "reason": "모델이 증거를 인용하지 못해 판정하지 않았다. " + reason}
        if verdict == "related" and not any(level[x] in DIRECT for x in cited):
            verdict = "weak"                                  # 중간 키워드를 거친 증거만으로는 직접 관계라 하지 않는다
        if verdict == "unrelated":
            typ = NONE
        return {"verdict": verdict, "type": typ, "direction": direction, "reason": reason, "evidence": cited}

    def _key(self, payload: dict) -> str:
        raw = json.dumps([self.llm.provider, self.llm.model, self.prompt_version, payload], ensure_ascii=False, sort_keys=True)
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
