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


@pytest.mark.parametrize("case", [c for c in load_cases() if c.get("needs") != "sound" and not c.get("auto_only")], ids=lambda c: c["id"])
def test_cases_with_manual_facts(case):
    got, _ = run_case(case)
    s = score_case(case, got)
    assert not s["mismatch"], s["mismatch"]


def _case(cid):
    return next(c for c in load_cases() if c["id"] == cid)


def test_explanation_is_friendly_and_defines_terms():
    got, _ = run_case(_case("dijkstra-fibonacci-amortized"))
    r2 = next(d for d in got if d.rule == "R2")
    e = r2.explanation
    assert e.startswith(r2.conclusion + ". 쉽게 말하면,") and "차근차근" in e
    assert "삽입은 새 원소를 넣는 일이에요" in e and "O(log n)이라는 건 n이 아주 커져도" in e   # 용어·표기 풀이
    assert "상각 시간은 연산 하나하나가 아니라" in e
    assert all(s.text in e for s in r2.steps)                                         # 계산 단계는 그대로 들어간다
    assert "이 결론이 성립하는 조건이에요." in e


def test_explanation_of_held_and_rejected_says_what_is_missing():
    got, _ = run_case(_case("decrease-key-cost-missing"))
    held = next(d for d in got if d.status == "held")
    assert held.explanation.startswith("아직 확정하지 못한 결론이에요") and "키 줄이기 연산 비용" in held.explanation
    assert "쉽게 말하면" not in held.explanation and "먼저 나오는 말" not in held.explanation
    got, _ = run_case(_case("plain-heap-no-decrease-key"))
    assert got[0].explanation.startswith("이 결론은 버렸어요") and "그 까닭은 이래요." in got[0].explanation


@pytest.mark.parametrize("expr, want", [
    ("1", "입력이 커져도 걸리는 시간이 거의 그대로라는 뜻이에요"), ("log n", "n이 아주 커져도 아주 천천히만 늘어난다는 뜻이에요"),
    ("V", "V에 비례해서 늘어난다는 뜻이에요"), ("V^2", "V가 두 배가 되면 대략 네 배가 된다는 뜻이에요"),
    ("(V + E) log V", None), ("foo(n)", None)])
def test_cost_gloss(expr, want):
    from app.services.derive.plain import cost_gloss
    assert cost_gloss(expr) == want


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
                     Fact("b", "op_cost", "Y", "y", ev, op="Teleport Now"),
                     Fact("c", "op_cost", "Z", "Bad Key", ev, op="insert")])
    assert {p.reason for p in fs.problems} == {"missing_op", "invalid_op", "invalid_key"} and not fs.facts


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
                              _raw("insert: O(log n)", op="Teleport Now")]})                   # key 모양이 아님 (스키마는 못 막는다)
    ex = FactExtractor(fake, tmp_path)
    got = ex.extract("note-1", 2, body, "sha", ["dijkstra"])
    assert [f.op for f in got.facts] == ["insert"] and got.facts[0].origin == "extracted"
    f = got.facts[0]
    assert (f.evidence.start, f.evidence.end) == locate(body, "insert: O(log n)") and f.role is None
    assert got.dropped == {"quote_not_in_note": 1, "invalid_op": 1}
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


def test_open_vocabulary_new_role_and_op_get_labels():
    """어휘 밖의 역할·연산도 key + 한국어 이름으로 도출에 쓰인다 (DB 인덱스 조회)."""
    ev = lambda n: FactEvidence(n, 1, "s", 0, 1, "x")
    facts = [Fact("a1", "needs_op", "인덱스 중첩 루프 조인", "inlj", ev("N1"), role="join_index", op="point_lookup",
                  role_label="조인 인덱스", op_label="점 조회", expr="R"),
             Fact("a2", "size_bound", "인덱스 중첩 루프 조인", "inlj", ev("N1"), role="join_index", expr="S"),
             Fact("a3", "extra_cost", "인덱스 중첩 루프 조인", "inlj", ev("N1"), role="join_index", expr="R"),
             Fact("b1", "implements", "B+ 트리 인덱스", "bplus", ev("N2"), role="join_index"),
             Fact("b2", "op_cost", "B+ 트리 인덱스", "bplus", ev("N2"), op="point_lookup", op_label="점 조회",
                  expr="log n", case="worst")]
    got = {d.rule: d for d in derive(facts)}
    assert got["R2"].status == "conclusion" and got["R2"].result == "O(R log S)" and "점 조회" in got["R2"].explanation
    assert got["R2"].role_label == "조인 인덱스"


def _ev(note):
    return FactEvidence(note, 1, "s", 0, 1, "x")


def _fx(i, kind, subject, note, **kw):
    return Fact(i, kind, subject, kw.pop("key", subject), _ev(note), **kw)


def test_phase_counts_are_summed():
    """단계마다 적힌 횟수는 합친다: k번 넣고, 병합마다 k-1번 넣으면 2k-1번 (하나만 고르면 전체보다 작게 잡는다)."""
    facts = [_fx("a1", "needs_op", "H", "N1", role="priority_queue", op="insert", expr="k"),
             Fact("a1b", "needs_op", "H", "H", FactEvidence("N1", 1, "s", 5, 6, "y"), role="priority_queue", op="insert", expr="k-1"),
             _fx("a2", "size_bound", "H", "N1", role="priority_queue", expr="k"),
             _fx("a3", "extra_cost", "H", "N1", role="priority_queue", expr="k"),
             _fx("b1", "implements", "Heap", "N2", role="priority_queue"),
             _fx("b2", "op_cost", "Heap", "N2", op="insert", expr="log n")]
    r2 = next(d for d in derive(facts) if d.rule == "R2")
    assert r2.status == "conclusion" and r2.result == "O(k log k)"
    assert {"a1", "a1b"} <= set(r2.premises) and "k + k-1" in r2.explanation


def test_r2_per_cost_case_and_single_refused_op_rejects():
    facts = [_fx("a1", "needs_op", "J", "N1", role="dictionary", op="insert", expr="N"),
             _fx("a2", "needs_op", "J", "N1", role="dictionary", op="search", expr="M"),
             _fx("a3", "size_bound", "J", "N1", role="dictionary", expr="N"),
             _fx("a4", "extra_cost", "J", "N1", role="dictionary", expr="N+M"),
             _fx("b0", "implements", "T", "N2", role="dictionary")]
    for op in ("insert", "search"):
        facts += [_fx(f"b-{op}-avg", "op_cost", "T", "N2", op=op, expr="1", case="average"),
                  _fx(f"b-{op}-w", "op_cost", "T", "N2", op=op, expr="n", case="worst")]
    out = {d.case: d for d in derive(facts) if d.rule == "R2"}
    assert out["average"].result == "O(M + N)" and out["worst"].result == "O((M + N) N)"   # 경우마다 따로
    refused = [_fx("a1", "needs_op", "D", "N1", role="spatial_index", op="range_query", expr="N"),
               _fx("b1", "op_cost", "K", "N2", op="range_query", negated=True)]
    d = derive(refused)
    assert [(x.rule, x.status) for x in d] == [("R1", "rejected")]                       # 연산 하나만 겹쳐도 기각은 낸다
