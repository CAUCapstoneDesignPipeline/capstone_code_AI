"""조건부 결론 도출: Big-O 대수, R1·R2 규칙, 사례 (docs/team/data), 사실 추출·soundness 판정 (가짜 LLM)."""

import dataclasses

import pytest

from app.services.derive import bigo
from app.services.derive.cases import load_cases, load_note, locate, manual_facts, run_case, score_case
from app.services.derive.extract_llm import FactExtractor
from app.services.derive.facts import Fact, FactEvidence, FactSet
from app.services.derive.korean import josa
from app.services.derive.rules import derive
from app.services.derive.sound_llm import Premise, SoundIn, SoundnessJudge, apply
from app.services.llm import LLMError, LLMResult


@pytest.mark.parametrize("text, want", [
    ("O(log n)", "O(log n)"), ("|V| + |E|", "O(V + E)"), ("n²", "O(n^2)"), ("V-1", "O(V)"),
    ("O(V log V + E log V)", "O((V + E) log V)"), ("\\log_2 n", "O(log n)"), ("logn", "O(log n)"),
    ("O(1)", "O(1)"), ("E α(V)", "O(E α(V))"), ("(V+E)^2", "O(V^2 + V·E + E^2)"), ("V + V log V", "O(V log V)"),
])
def test_bigo_parse_and_show(text, want):
    assert bigo.big_o(bigo.parse(text)) == want


@pytest.mark.parametrize("text", ["2^n", "E/V", "Ω(n)", "n^k", "", "foo(n)"])
def test_bigo_rejects_what_it_cannot_bound(text):
    with pytest.raises(bigo.ParseError):
        bigo.parse(text)


def test_bigo_substitute_and_equivalent():
    per = bigo.substitute(bigo.parse("log n"), "n", bigo.parse("V"))
    total = bigo.add(bigo.mul(bigo.var("V"), per), bigo.mul(bigo.var("E"), per), bigo.parse("V+E"))
    assert bigo.big_o(total) == "O((V + E) log V)"
    assert bigo.equivalent(total, bigo.parse("(V+E) log V"))
    assert not bigo.equivalent(total, bigo.parse("E log V"))          # E ≥ V 를 가정하지 않는다
    assert bigo.big_o(bigo.substitute(bigo.parse("log n"), "n", bigo.parse("V^2"))) == "O(log V)"


def test_josa():
    assert josa("힙", "을") == "힙을" and josa("큐", "을") == "큐를" and josa("힙", "으로") == "힙으로"
    assert josa("큐", "으로") == "큐로" and josa("레벨", "으로") == "레벨로" and josa("O(log n)", "예요") == "O(log n)이에요"
    assert josa("O(log V)", "예요") == "O(log V)예요" and josa("BFS", "은") == "BFS는"


@pytest.mark.parametrize("case", [c for c in load_cases() if c.get("needs") != "sound"], ids=lambda c: c["id"])
def test_cases_with_manual_facts(case):
    got, _ = run_case(case)
    s = score_case(case, got)
    assert not s["mismatch"], s["mismatch"]


def _case(cid):
    return next(c for c in load_cases() if c["id"] == cid)


def test_dijkstra_explanation_cites_both_notes():
    got, facts = run_case(_case("dijkstra-indexed-heap"))
    r2 = next(d for d in got if d.rule == "R2")
    assert r2.result == "O((V + E) log V)" and r2.case == "worst" and len(r2.notes) == 2
    assert "「다익스트라 (Dijkstra) 정리」" in r2.explanation and "인덱스 있는 이진 최소 힙" in r2.explanation
    assert any("음수가 아님" in c for c, _ in r2.conditions)          # 원문 조건이 결론 조건으로 따라온다
    assert any("가장 빠른 구현이라는 뜻은 아니에요" in c for c, _ in r2.conditions)
    by_id = {f.fact_id: f for f in facts}
    for x in r2.premises:                                              # 전제마다 원문 위치가 quote 와 같다
        f = by_id[x]
        body = load_note(next(n for n in ("dijkstra", "indexed-heap") if load_note(n).note_id == f.note)).body
        assert body.encode("utf-16-le")[2 * f.evidence.start:2 * f.evidence.end].decode("utf-16-le") == f.evidence.quote


def test_missing_premise_holds_only_dependent_conclusion():
    got, _ = run_case(_case("decrease-key-cost-missing"))
    st = {(d.rule, d.algorithm[0]): d for d in got}
    assert st[("R2", "dijkstra")].status == "held" and st[("R2", "dijkstra")].missing
    assert st[("R1", "dijkstra")].status == "conclusion"            # 독립 결론은 남는다
    assert st[("R2", "bfs")].status == "conclusion"


def test_same_note_premises_are_not_new():
    """두 노트가 서로 다른 필수 전제를 내야 한다: 한 노트에 모두 있으면 known."""
    note = load_note("dijkstra")
    facts = manual_facts(note) + [dataclasses.replace(f, evidence=dataclasses.replace(f.evidence, note_id=note.note_id))
                                  for f in manual_facts(load_note("indexed-heap"))]
    assert {d.status for d in derive(facts)} == {"known"}


def test_derive_is_deterministic():
    case = _case("heap-contradiction")
    a, _ = run_case(case)
    b, _ = run_case(case)
    assert [(d.derivation_id, d.status, d.result) for d in a] == [(d.derivation_id, d.status, d.result) for d in b]


def test_unreadable_expression_holds():
    note = load_note("dijkstra")
    facts = [dataclasses.replace(f, expr="2^n") if f.op == "insert" and f.kind == "needs_op" else f
             for f in manual_facts(note)] + manual_facts(load_note("indexed-heap"))
    r2 = next(d for d in derive(facts) if d.rule == "R2")
    assert r2.status == "held" and any("식을 읽지 못했어요" in r for r in r2.reasons)


def test_factset_drops_malformed():
    ev = FactEvidence("n", 1, "s", 0, 1, "x")
    fs = FactSet.of([Fact("a", "needs_op", "X", "x", ev, role="priority_queue"),          # op 없음
                     Fact("b", "op_cost", "Y", "y", ev, op="teleport"),
                     Fact("c", "op_cost", "Z", "Bad Key", ev, op="insert")])
    assert {p.reason for p in fs.problems} == {"missing_op", "unknown_op", "invalid_key"} and not fs.facts


# ---- LLM (가짜) ----

class FakeLLM:
    provider, model = "anthropic", "claude-opus-5-5"

    def __init__(self, out=None, error=None):
        self.out, self.error, self.calls = out, error, []

    def complete_json(self, system, user, schema, name):
        self.calls.append((system, user, schema))
        if self.error:
            raise LLMError(self.error)
        return LLMResult(self.out(user) if callable(self.out) else self.out, self.provider, self.model)


def _raw(quote, **kw):
    base = {"kind": "op_cost", "subject": "힙", "subjectKey": "heap", "role": "none", "op": "insert", "expr": "log n",
            "case": "worst", "negated": False, "condition": "", "other": "", "otherKey": "", "quote": quote}
    return {**base, **kw}


def test_extractor_grounds_quotes_and_drops_invented(tmp_path):
    body = "# 힙\n- insert: O(log n)\n"
    fake = FakeLLM({"facts": [_raw("insert: O(log n)"),
                              _raw("extract-min: O(log n)", op="extract_min"),                 # 원문에 없는 quote
                              _raw("insert: O(log n)", op="teleport")]})                       # enum 밖 (스키마가 막지만 다시 본다)
    ex = FactExtractor(fake, tmp_path)
    got = ex.extract("note-1", 2, body, "sha", ["dijkstra"])
    assert [f.op for f in got.facts] == ["insert"] and got.facts[0].origin == "extracted"
    f = got.facts[0]
    assert (f.evidence.start, f.evidence.end) == locate(body, "insert: O(log n)") and f.role is None
    assert got.dropped == {"quote_not_in_note": 1, "unknown_op": 1}
    assert "dijkstra" in fake.calls[0][1]                               # 이미 있는 이름을 알려 준다
    ex.extract("note-1", 2, body, "sha", ["dijkstra"])
    assert len(fake.calls) == 1                                         # 캐시


def test_extractor_refusal_means_no_facts(tmp_path):
    got = FactExtractor(FakeLLM(error="refusal"), tmp_path).extract("n", 1, "body", "s")
    assert got.facts == []
    with pytest.raises(LLMError):
        FactExtractor(FakeLLM(error="timeout"), tmp_path / "x").extract("n", 1, "body", "s")


def _sound_items():
    got, facts = run_case(_case("dijkstra-indexed-heap"))
    by_id = {f.fact_id: f for f in facts}
    items = [SoundIn(d.derivation_id, d.conclusion, tuple(c for c, _ in d.conditions), tuple(s.text for s in d.steps),
                     tuple(Premise(x, by_id[x].text(), by_id[x].evidence.quote) for x in d.premises)) for d in got]
    return got, by_id, items


def test_soundness_rejects_and_propagates_to_r2(tmp_path):
    got, by_id, items = _sound_items()

    def out(user):          # D1 = R1 은 P1 이 충실하지 않다, D2 = R2 는 sound
        return {"judgments": [{"derivation": "D1", "unfaithful": ["P1"], "verdict": "sound", "reason": "근거가 다릅니다."},
                              {"derivation": "D2", "unfaithful": [], "verdict": "sound", "reason": "맞습니다."}]}
    verdicts, _, _ = SoundnessJudge(FakeLLM(out), tmp_path).judge(items)
    apply(got, {v.ref: v for v in verdicts}, by_id)
    assert [d.status for d in got] == ["rejected", "rejected"]          # R1 이 버려지면 R2 도
    assert "근거 원문이 이 사실을 말하지 않아요" in got[0].explanation


def test_soundness_uncertain_holds_and_bad_output_is_error(tmp_path):
    got, by_id, items = _sound_items()
    ok = {"judgments": [{"derivation": "D1", "unfaithful": [], "verdict": "sound", "reason": "맞습니다."},
                        {"derivation": "D2", "unfaithful": [], "verdict": "uncertain", "reason": "모르겠습니다."}]}
    verdicts, _, _ = SoundnessJudge(FakeLLM(ok), tmp_path).judge(items)
    apply(got, {v.ref: v for v in verdicts}, by_id)
    assert [d.status for d in got] == ["conclusion", "held"]
    bad = {"judgments": [{"derivation": "D1", "unfaithful": ["P99"], "verdict": "sound", "reason": "x"}]}
    with pytest.raises(LLMError):
        SoundnessJudge(FakeLLM(bad), tmp_path / "b").judge(items)
