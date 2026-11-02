"""도출 사례 (docs/team/data): 노트·사람이 적은 사실·기대 결과를 읽고, 도출을 돌려 기대와 비교한다.

  docs/team/data/notes/<이름>.md      노트 원문
  docs/team/data/facts/<이름>.json    그 노트에서 사람이 적은 사실 (quote 는 원문 그대로, 위치는 여기서 찾는다)
  docs/team/data/cases.json           사례: 쓰는 노트들과 기대 결과 (rule·algorithm·structure·status·result)

점수 (안건 6 의 지표를 사례 단위로)
  정밀도         시스템이 conclusion 으로 낸 것 중 기대에도 conclusion 인 비율
  재현율         기대 conclusion 중 시스템이 conclusion 으로 낸 비율 (결과식이 있으면 식까지 같아야 한다)
  잘못된 확정    기대 conclusion 이 없는 사례(음성) 중 시스템이 conclusion 을 하나라도 낸 사례 비율
  상태 일치      기대에 적힌 (rule, algorithm, structure) 마다 상태가 같은 비율 (보류·기각·이미 앎까지)
"""

import json
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from .facts import Fact, FactEvidence, subject_key
from .rules import Derivation, derive

DATA = Path(__file__).resolve().parents[3] / "docs/team/data"
_NS = uuid.UUID("5b8f0a52-8f0e-4d5c-9a51-7b0e0c6f3a10")       # app/client.py 의 uid 와 같은 이름 공간


def note_uuid(name: str) -> str:
    return str(uuid.uuid5(_NS, "\x1f".join(("note", name))))


def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def locate(body: str, quote: str) -> tuple[int, int] | None:
    """quote 의 첫 위치 (UTF-16). 없으면 None."""
    i = body.find(quote)
    if i < 0:
        return None
    start = utf16_len(body[:i])
    return start, start + utf16_len(quote)


def fact_from_dict(d: dict, fact_id: str, note_id: str, note_version: int, body: str, body_sha: str,
                   origin: str) -> Fact | None:
    """사람이 적은 사실(quote 로 위치를 찾는다) → Fact. quote 가 원문에 없으면 None."""
    span = locate(body, d["quote"])
    if span is None:
        return None
    return Fact(fact_id, d["kind"], d["subject"], d.get("subjectKey") or subject_key(d["subject"]),
                FactEvidence(note_id, note_version, body_sha, span[0], span[1], d["quote"]),
                role=d.get("role"), op=d.get("op"), role_label=d.get("roleLabel"), op_label=d.get("opLabel"),
                expr=d.get("expr"), case=d.get("case"),
                negated=bool(d.get("negated", False)), condition=d.get("condition"), other=d.get("other"),
                other_key=d.get("otherKey"), origin=origin)


@dataclass
class Note:
    name: str
    note_id: str
    title: str
    body: str
    sha: str


def load_note(name: str, data: Path = DATA) -> Note:
    import hashlib
    body = unicodedata.normalize("NFC", (data / "notes" / f"{name}.md").read_text(encoding="utf-8"))
    title = next((l.lstrip("# ").strip() for l in body.splitlines() if l.startswith("#")), name)
    return Note(name, note_uuid(name), title, body, hashlib.sha256(body.encode()).hexdigest())


def manual_facts(note: Note, data: Path = DATA) -> list[Fact]:
    path = data / "facts" / f"{note.name}.json"
    if not path.exists():
        return []
    out = []
    for i, d in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        f = fact_from_dict(d, f"{note.name}#{i + 1}", note.note_id, 1, note.body, note.sha, "manual")
        if f is None:
            raise ValueError(f"{note.name}: 원문에 없는 quote {d['quote']!r}")
        out.append(f)
    return out


def load_cases(data: Path = DATA) -> list[dict]:
    """사례 목록. cases.json 의 aliases (기대 key → LLM 이 달리 쓸 수 있는 key) 를 사례마다 붙인다."""
    raw = json.loads((data / "cases.json").read_text(encoding="utf-8"))
    cases = [{**c, "aliases": raw.get("aliases", {})} for c in raw["cases"]]
    test = data / "cases-test.json"                      # 독립 작성한 최종 시험. 고칠 때는 보지 않는다 (--split test)
    if test.exists():
        cases += [{**c, "split": "test", "aliases": {}} for c in json.loads(test.read_text(encoding="utf-8"))["cases"]]
    return cases


def _match(case: dict, got: list[Derivation]) -> tuple[dict, dict]:
    """(기대 key → 기대, key → 도출). 기대 두 가지 쓰기:
      - 키로: (rule, algorithm, structure). 주체 key 는 별칭을 기대 key 로 바꿔 비교한다 (LLM 추출은 같은 대상을
        binary_min_heap 처럼 달리 부를 수 있다)
      - 노트로 (독립 작성한 사례): {"rule", "notes": [전제를 낸 노트 둘]}. 키 없이 (규칙, 노트 쌍)으로 짝짓고,
        같은 노트 쌍에 도출이 여럿이면 기대한 상태와 같은 것을 먼저 고른다
    기대에 없는 도출은 따로 키를 달아 have 에만 넣는다 (잘못된 확정을 세려고)."""
    canon = {a: k for k, alts in case.get("aliases", {}).items() for a in alts}
    uid = {n: note_uuid(n) for n in case["notes"]}
    back = {v: k for k, v in uid.items()}
    want, have, used = {}, {}, set()
    for i, e in enumerate(case["expect"]):
        if "notes" in e:
            key = (e["rule"], "+".join(sorted(e["notes"])), f"#{i}")
            want[key] = e
            ids = frozenset(uid[n] for n in e["notes"])
            # 기대한 노트 둘을 모두 쓴 도출 (제3 노트가 같은 말을 되풀이해 셋이 되어도 인정, 노트가 적은 쪽을 먼저)
            hits = sorted((d for d in got if d.rule == e["rule"] and ids <= frozenset(d.notes) and d.derivation_id not in used),
                          key=lambda d: len(d.notes))
            fits = lambda d: d.status == e["status"] and e.get("case", d.case) == d.case and e.get("result", d.result) == d.result
            hit = next((d for d in hits if fits(d)), next((d for d in hits if d.status == e["status"]),
                                                          hits[0] if hits else None))
            if hit is not None:
                have[key] = hit
                used.add(hit.derivation_id)
        else:
            want[(e["rule"], e["algorithm"], e["structure"])] = e
    for d in got:
        if d.derivation_id in used:
            continue
        key = (d.rule, canon.get(d.algorithm[0], d.algorithm[0]), canon.get(d.structure[0], d.structure[0]))
        if key in want and key not in have:
            have[key] = d
        else:
            have[(d.rule, "+".join(sorted(back.get(n, n) for n in d.notes)), "+" + d.derivation_id)] = d
        used.add(d.derivation_id)
    return want, have


def score_case(case: dict, got: list[Derivation]) -> dict:
    """사례 하나의 비교. got 은 soundness 까지 끝난 최종 상태여야 한다."""
    want, have = _match(case, got)
    concluded = [k for k, d in have.items() if d.status == "conclusion"]
    expected = [k for k, e in want.items() if e["status"] == "conclusion"]

    def right(k):
        e, d = want.get(k), have.get(k)
        if not e or not d or e["status"] != d.status:
            return False
        return ("result" not in e or e["result"] == d.result) and ("case" not in e or e["case"] == d.case)

    status_ok = [k for k in want if have.get(k) is not None and have[k].status == want[k]["status"]
                 or (want[k]["status"] == "absent" and k not in have)]
    return {"id": case["id"], "split": case.get("split", "dev"),
            "true_conclusions": sum(1 for k in concluded if right(k)), "conclusions": len(concluded),
            "found": sum(1 for k in expected if right(k)), "expected": len(expected),
            "negative": not expected, "false_confirm": bool(concluded) and not expected,
            "status_ok": len(status_ok), "status_total": len(want),
            "mismatch": [{"key": "/".join(k), "want": want[k]["status"] if k in want else "-",
                          "got": have[k].status if k in have else "-",
                          "result": have[k].result if k in have else None}
                         for k in sorted(set(want) | set(concluded)) if not right(k) and
                         not (k in want and want[k]["status"] != "conclusion" and have.get(k) and
                              have[k].status == want[k]["status"])]}


def run_case(case: dict, data: Path = DATA, facts_of=None) -> tuple[list[Derivation], list[Fact]]:
    """사례의 노트들로 도출 (결과, 쓴 사실). facts_of(note) 가 없으면 사람이 적은 사실을 쓴다."""
    notes = [load_note(n, data) for n in case["notes"]]
    facts = [f for n in notes for f in (facts_of(n) if facts_of else manual_facts(n, data))]
    titles = {n.note_id: n.title for n in notes}
    return derive(facts, titles), facts
