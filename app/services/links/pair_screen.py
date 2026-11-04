"""후보 쌍이 합쳐서 새 지식이 될 만한지 LLM 으로 거른다 (선택). 외부 지식을 찾기 전에 쓴다.

두 문장(A, B)과 일반 상식으로 둘 어느 쪽에도 없는 구체적이고 맞는 결론이 나오는지를 다섯 라벨로 묻는다.
  useful · vague · redundant · unrelated · unusable
실제 노트 275쌍에서 AI 라벨러와 비교했을 때 useful 로 고른 쌍의 35~44% 가 라벨러도 useful 이라 했다 (거르지 않으면 11%).
라벨러가 useful 이라 한 것의 40~50% 만 찾는다 — 놓치는 쌍이 있다. 자동 재시도는 없고, 빠뜨린 쌍만 작은 묶음으로 한 번 더 묻는다.
결과는 hash(제공자, 모델, 프롬프트 버전, 두 문장) 로 디스크에 캐싱한다.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT_VERSION = "pairscreen-llm-1"
LABELS = ["useful", "vague", "redundant", "unrelated", "unusable"]
UNSURE = "unsure"

SYSTEM = """You screen pairs of statements taken from a student's Korean/English lecture notes (two different notes, two different keywords).
For EACH pair decide whether combining statement A and statement B (plus ordinary well-established general knowledge) can produce a NEW piece of knowledge for the student. Choose exactly one label:
- useful: combining them yields a specific, correct conclusion that is stated in neither statement alone and is non-trivial (e.g. a chain of reasoning A->B->C, a meaningful comparison/contrast, applying a general principle from one to the case in the other, one supplying a missing condition/step for the other).
- vague: they are about related things but the only thing you can say is generic ("both concern X", "X is related to Y"); no specific conclusion.
- redundant: one statement already says (essentially) what the other says or what the combination would conclude; nothing new.
- unrelated: no meaningful connection (shared word only, different senses of a word, different subjects).
- unusable: a statement is a fragment / cannot be understood on its own (e.g. a bare heading, a table cell, a title, a course-admin line), so no judgment is possible.
If the label is useful, also give the new conclusion in ONE short sentence (Korean); if it would be factually wrong or you are unsure it is correct, do not label useful. Be strict: when in doubt between useful and vague, choose vague. Return one result per id."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["results"], "properties": {"results": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "label", "conclusion"],
    "properties": {"id": {"type": "string"}, "label": {"type": "string", "enum": LABELS}, "conclusion": {"type": "string"}}}}}}


class PairScreen:
    def __init__(self, llm, cache_dir: Path | None, batch: int = 10):
        self.llm, self.cache_dir, self.batch = llm, cache_dir, batch
        self.prompt_version = PROMPT_VERSION

    @property
    def model(self) -> str:
        return self.llm.model

    def screen(self, pairs: list[dict]) -> dict[str, dict]:
        """pairs: [{id, a: {keyword, text}, b: {keyword, text}}] → id → {label, conclusion, cached}.
        모델이 빠뜨린 쌍은 한 번만 더 묻고, 그래도 없으면 label 이 unsure 다."""
        out: dict[str, dict] = {}
        todo = []
        for p in pairs:
            hit = self._cache_get(self._key(p))
            if hit is not None:
                out[p["id"]] = {**hit, "cached": True}
            else:
                todo.append(p)
        for size in (self.batch, 3):
            if not todo:
                break
            chunks = [todo[i:i + size] for i in range(0, len(todo), size)]
            with ThreadPoolExecutor(min(4, len(chunks))) as ex:
                for part in ex.map(self._call, chunks):
                    for p, res in part:
                        if res is not None:
                            self._cache_put(self._key(p), res)
                            out[p["id"]] = {**res, "cached": False}
            todo = [p for p in todo if p["id"] not in out]
        for p in todo:
            out[p["id"]] = {"label": UNSURE, "conclusion": "", "cached": False}
        return out

    def _call(self, chunk: list[dict]):
        user = "\n\n".join(f"id: {p['id']}\nA [{p['a']['keyword']}]: {p['a']['text']}\nB [{p['b']['keyword']}]: {p['b']['text']}" for p in chunk)
        data = self.llm.complete_json(SYSTEM, user, SCHEMA, "pair_screen").data
        got = {r["id"]: r for r in data.get("results", []) if isinstance(r, dict) and r.get("label") in LABELS}
        return [(p, ({"label": got[p["id"]]["label"], "conclusion": str(got[p["id"]].get("conclusion", "")).strip()}
                     if p["id"] in got else None)) for p in chunk]

    def _key(self, p: dict) -> str:
        raw = json.dumps([self.llm.provider, self.llm.model, self.prompt_version, p["a"]["keyword"], p["a"]["text"],
                          p["b"]["keyword"], p["b"]["text"]], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str):
        if self.cache_dir and (self.cache_dir / f"{key}.json").exists():
            return json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, value: dict):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
