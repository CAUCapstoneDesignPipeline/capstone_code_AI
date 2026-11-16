"""빠른 관계 추출 (`pairs-1`): 후보 쌍마다 Decisions 객관식 확률 → 학습한 트리 분류기 → 관계. 규칙·구조 관계는 쓰지 않는다 (정답 관계를 거의 못 찾는다, docs/eval/relations.md).

  1. 후보 쌍 (candidates.py): 같은 문장 ∪ 같은 절 ∪ 제목↔절 안의 개념 쌍. 한도를 넘으면 같은 문장·제목 쌍과 가까운 쌍을 먼저 남긴다 (MAX_PAIRS).
  2. 판정: 노트 원문과 쌍마다의 객관식 질문 하나 (관계 없음 / 유형 × 방향 / 대칭 유형 15개 선택지) 를 요청 하나(400쌍까지)로 보낸다. 쌍 수가 늘어도 지연은 거의 그대로다.
  3. 분류: 쌍의 구조·위치 특징 + 15개 선택지 확률 → 트리 둘 (관계가 있는가 / 있다면 어떤 유형·방향인가, `pairs-1.json`, numpy 로 계산). 같은 입력·같은 답이면 항상 같은 결과다.
  4. 근거: 두 개념이 가장 가까이 나온 곳의 문장 (제목·표 머리면 그 줄). 원문에서 그대로 자르므로 quote 는 항상 원문 slice 다.

학습: `experiments/relations/train_pair.py` (교사 = 같은 노트를 Claude Sonnet 이 읽고 낸 관계). 측정과 한계: docs/eval/relations.md.
질문 문구·선택지·특징·임계값은 profile 의 일부라 바꾸면 PROMPT_VERSION 과 profile id 를 함께 올린다. 노트 원문이 OpenAI 로 나간다.
"""

import bisect
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..concepts.judge import create_answers
from .candidates import candidate_pairs
from .extract import Evidence, Limits, Relation, _py_offsets, finalize
from .rules import code_for
from .sentences import sentence_spans

PROMPT_VERSION = "pairs-1"
MAX_QUESTIONS = 400             # 요청 하나의 질문 수 상한 (Decisions API)
MAX_PAIRS = 800                 # 노트 하나에서 판정하는 쌍 상한 (요청 두 번). 개념이 아주 많은 노트의 비용을 묶는다

SYM = ("contrasts_with", "analogous_to")
CLASSES = ["is_a_ab", "is_a_ba", "part_of_ab", "part_of_ba", "causes_ab", "causes_ba", "requires_ab", "requires_ba", "used_for_ab", "used_for_ba",
           "example_of_ab", "example_of_ba", "contrasts_with", "analogous_to"]
CUES = ["은", "는", "이다", "입니다", "한다", "된다", "의", "와", "과", "및", "또는", "→", "->", ":", "(", ")", ",", "를", "을", "에서", "으로", "로", "위해", "때문", "필요", "달리", "반면", "예", "등", "처럼", "vs", "VS", "|", "-"]

QUESTION = ('Read the study note and decide what relation the NOTE ITSELF states between the two terms "{a}" (A) and "{b}" (B), in the sentence, list, table or heading structure where they appear. '
            'Use only what the note says, never outside knowledge. Mere co-occurrence, listing, or sitting in the same section is NOT a relation: choose "none" then.')
CHOICES = [("none", "the note states no relation between A and B"),
           ("is_a_ab", "A is a kind of B (A is a subtype or specific form of B)"), ("is_a_ba", "B is a kind of A"),
           ("part_of_ab", "A is a component or part of B"), ("part_of_ba", "B is a component or part of A"),
           ("causes_ab", "A causes, produces or leads to B"), ("causes_ba", "B causes, produces or leads to A"),
           ("requires_ab", "A needs B in order to hold or work"), ("requires_ba", "B needs A in order to hold or work"),
           ("used_for_ab", "A is used for B (B is the purpose or goal of A)"), ("used_for_ba", "B is used for A (A is the purpose or goal of B)"),
           ("example_of_ab", "A is a concrete example or instance of B"), ("example_of_ba", "B is a concrete example or instance of A"),
           ("contrasts_with", "the note contrasts or compares A and B (neither is the subject)"),
           ("analogous_to", "A and B work on a similar principle")]
FEATURES = (["kind_sent", "kind_section", "kind_head", "sent_dist", "both_sent", "both_line", "char_dist", "between_len", "head_a", "head_b", "head_min_a", "head_min_b",
             "n_a", "n_b", "n_pairs", "first_a", "first_b", "count_ratio", "list_a", "list_b", "table_a", "table_b", "a_first", "len_a", "len_b",
             "latin_a", "latin_b", "hangul_a", "hangul_b", "contained", "paren_a", "paren_b", "bold_a", "bold_b"]
            + [f"between:{c}" for c in CUES] + ["d_related"] + [f"d_{c}" for c in CLASSES])
N_STRUCT = len(FEATURES) - 1 - len(CLASSES)


def named_choices(a: str, b: str) -> list[dict]:
    """선택지 설명에 A·B 대신 실제 표기를 넣는다."""
    return [{"value": v, "description": d.replace("A and B", f'"{a}" and "{b}"').replace("A is", f'"{a}" is').replace("B is", f'"{b}" is').replace("A needs", f'"{a}" needs').replace("B needs", f'"{b}" needs').replace("A causes", f'"{a}" causes').replace("B causes", f'"{b}" causes').replace("A work", f'"{a}" work').replace(" of B", f' of "{b}"').replace(" of A", f' of "{a}"').replace("for B", f'for "{b}"').replace("for A", f'for "{a}"').replace("in B", f'in "{b}"').replace("to B", f'to "{b}"').replace("to A", f'to "{a}"')} for v, d in CHOICES]


def decode(value: str):
    """선택지 값 → (유형, a 가 주어인가) 또는 None."""
    if value == "none":
        return None
    if value in SYM:
        return value, True
    t, d = value.rsplit("_", 1)
    return t, d == "ab"


def positions(text: str, mentions: dict) -> dict[str, list[tuple[int, int]]]:
    """개념 key → 언급 위치 (Python 문자) 목록, 앞에서부터."""
    P: dict = {}
    for o in _py_offsets(text, mentions):
        P.setdefault(o.key, []).append((o.start, o.end))
    return P


def _nearest(pa, pb):
    """두 개념의 언급 중 가장 가까운 쌍 → (거리, 시작a, 끝a, 시작b, 끝b)."""
    return min(((abs(s2 - e1) if s2 >= e1 else abs(s1 - e2)), s1, e1, s2, e2) for s1, e1 in pa for s2, e2 in pb)


def structure_features(text: str, P: dict, pairs) -> np.ndarray:
    """쌍마다 구조·위치 특징 (FEATURES 앞 N_STRUCT 개)."""
    sents = sentence_spans(text)
    s_starts = [s for s, _ in sents]
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line), line))
        pos += len(line) + 1
    l_starts = [s for s, _, _ in lines]

    def sent_of(p):
        i = bisect.bisect_right(s_starts, p) - 1
        return i if i >= 0 and p < sents[i][1] else -1

    def line_of(p):
        i = bisect.bisect_right(l_starts, p) - 1
        return i if i >= 0 and p <= lines[i][1] else -1

    head_level = lambda i: (len(re.match(r"^#+", lines[i][2]).group(0)) if i >= 0 and re.match(r"^#+\s", lines[i][2]) else 0)
    is_list = lambda i: i >= 0 and bool(re.match(r"^\s*([-*+]|\d+[.)])\s", lines[i][2]))
    is_table = lambda i: i >= 0 and lines[i][2].lstrip().startswith("|")
    n_chars = max(1, len(text))
    X = []
    for a, b, kinds in pairs:
        pa, pb = P[a], P[b]
        sa, sb = [sent_of(s) for s, _ in pa], [sent_of(s) for s, _ in pb]
        la, lb = [line_of(s) for s, _ in pa], [line_of(s) for s, _ in pb]
        cd, s1, e1, s2, e2 = _nearest(pa, pb)
        between = text[min(e1, e2):max(s1, s2)][:80] if max(s1, s2) > min(e1, e2) else ""
        ha, hb = [head_level(i) for i in la], [head_level(i) for i in lb]
        f = [("sent" in kinds), ("section" in kinds), ("head" in kinds), min(abs(x - y) for x in sa for y in sb), len(set(sa) & set(sb)), len(set(la) & set(lb)),
             min(cd, 400), len(between), max(ha), max(hb), min(h for h in ha if h) if any(ha) else 0, min(h for h in hb if h) if any(hb) else 0,
             len(pa), len(pb), len(pa) * len(pb), pa[0][0] / n_chars, pb[0][0] / n_chars, min(len(pa), len(pb)) / max(len(pa), len(pb)),
             any(is_list(i) for i in la), any(is_list(i) for i in lb), any(is_table(i) for i in la), any(is_table(i) for i in lb),
             pa[0][0] < pb[0][0], len(a), len(b), bool(re.search(r"[A-Za-z]", a)), bool(re.search(r"[A-Za-z]", b)), bool(re.search(r"[가-힣]", a)), bool(re.search(r"[가-힣]", b)),
             a.lower() in b.lower() or b.lower() in a.lower(), text[e1:e1 + 1] in ("(", "（"), text[e2:e2 + 1] in ("(", "（"),
             s1 < s2 and (re.search(r"\*\*", text[max(0, s1 - 2):s1]) is not None), (re.search(r"\*\*", text[max(0, s2 - 2):s2]) is not None)]
        X.append(f + [between.count(c) for c in CUES])
    return np.array(X, float).reshape(len(pairs), N_STRUCT)


def decision_features(pairs, decisions) -> np.ndarray:
    """쌍마다 [1 - P(관계 없음), 선택지 14개 확률]. 답이 없는 쌍(거절)은 빈 dict 로 본다."""
    rows = []
    for a, b, _ in pairs:
        d = decisions[(a, b)] or {}
        rows.append([1.0 - d.get("none", 0.0)] + [d.get(c, 0.0) for c in CLASSES])
    return np.array(rows, float).reshape(len(pairs), 1 + len(CLASSES))


class Forest:
    """부스팅 트리 묶음 (트리 × 후보를 한꺼번에 내려가며 잎 값을 더한다). export 형식은 experiments/relations/train_pair.py 가 쓴다."""

    def __init__(self, trees: list[dict], baseline: float, depth: int):
        width = max(len(t["feature"]) for t in trees)
        pad = lambda key, dtype, fill: np.array([t[key] + [fill] * (width - len(t[key])) for t in trees], dtype=dtype)
        self.feature, self.cut = pad("feature", np.int64, 0), pad("threshold", np.float64, 0.0)
        self.left, self.right, self.value = pad("left", np.int64, -1), pad("right", np.int64, -1), pad("value", np.float64, 0.0)
        self.baseline, self.depth = float(baseline), int(depth)

    def raw(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        T = len(self.feature)
        node = np.zeros((T, n), dtype=np.int64)
        cols, tree = np.arange(n)[None, :], np.arange(T)[:, None]
        for _ in range(self.depth):
            left = self.left[tree, node]
            go_left = X[cols, self.feature[tree, node]] <= self.cut[tree, node]
            node = np.where(left < 0, node, np.where(go_left, left, self.right[tree, node]))
        return self.baseline + self.value[tree, node].sum(0)


class PairModel:
    """쌍 분류기 둘: 관계가 있는가 (확률), 있다면 어느 유형·방향인가 (CLASSES 번호)."""

    def __init__(self, data: dict):
        if list(data["features"]) != FEATURES or list(data["classes"]) != CLASSES:
            raise ValueError("모델 파일의 특징·클래스가 코드와 다르다 — 다시 학습해야 한다")
        self.id, self.threshold = data["id"], float(data["threshold"])
        self.pair = Forest(data["pair"]["trees"], data["pair"]["baseline"], data["pair"]["max_depth"])
        self.type_classes = np.array(data["type"]["classes"], dtype=np.int64)
        self.types = [Forest(t["trees"], t["baseline"], t["max_depth"]) for t in data["type"]["per_class"]]
        self.trained = data.get("trained", {})
        self.version = f"{self.id}+{hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]}"

    @classmethod
    def load(cls, name: str = "pairs-1") -> "PairModel":
        return cls(json.loads((Path(__file__).parent / f"{name}.json").read_text(encoding="utf-8")))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """→ (관계 확률, 유형·방향 CLASSES 번호). 플랫폼별 부동소수 끝자리가 경계에서 갈리지 않게 확률은 6자리로 둔다."""
        if len(X) == 0:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        p = np.round(1.0 / (1.0 + np.exp(-self.pair.raw(X))), 6)
        scores = np.stack([f.raw(X) for f in self.types], axis=1)
        return p, self.type_classes[scores.argmax(1)]


class DecisionsPairs:
    """쌍마다 객관식 확률 dict 를 돌려준다. client 는 openai.OpenAI 처럼 decisions.create 가 있는 것."""

    provider = "openai"

    def __init__(self, client, model: str):
        self.client, self.model = client, model

    def ask(self, text: str, pairs) -> list[dict | None]:
        labels = [(a, b) for a, b, _ in pairs]
        chunks = [labels[i:i + MAX_QUESTIONS] for i in range(0, len(labels), MAX_QUESTIONS)]

        def call(chunk):
            qs = [{"type": "choice", "instructions": QUESTION.format(a=a, b=b), "choices": named_choices(a, b)} for a, b in chunk]
            answers = create_answers(self.client, self.model, text, qs)
            return [None if a.type == "refusal" else {x.value: x.probability for x in a.probabilities} for a in answers]

        if not chunks:
            return []
        with ThreadPoolExecutor(len(chunks)) as pool:
            return [p for out in pool.map(call, chunks) for p in out]


def _evidence_spans(text: str, pa, pb, sents, starts) -> set[tuple[int, int]]:
    """두 개념이 같은 문장에 나온 곳 중 가장 가까운 곳의 그 문장. 같은 문장이 없으면 가장 가까운 두 언급이 각각 든 문장(줄)."""
    sent_of = lambda pos: (lambda i: i if i >= 0 and pos < sents[i][1] else -1)(bisect.bisect_right(starts, pos) - 1)
    together = [(abs(s2 - e1) if s2 >= e1 else abs(s1 - e2), s1, e1, s2, e2) for s1, e1 in pa for s2, e2 in pb if sent_of(s1) == sent_of(s2) != -1]
    _, s1, _, s2, _ = min(together) if together else _nearest(pa, pb)
    return {_span_at(text, s1, sents, starts), _span_at(text, s2, sents, starts)}


def _span_at(text: str, pos: int, sents, starts) -> tuple[int, int]:
    """언급이 든 문장. 문장이 아닌 줄(제목·표 머리)이면 그 줄."""
    i = bisect.bisect_right(starts, pos) - 1
    if i >= 0 and pos < sents[i][1]:
        return sents[i]
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    e = len(text) if e < 0 else e
    seg = text[s:e]
    return s + (len(seg) - len(seg.lstrip())), s + len(seg.rstrip())


class PairRelations:
    """후보 쌍 → Decisions → 분류기 → 관계·근거. extract 의 반환은 extract_relations 와 같은 모양이다."""

    provider = "openai"

    def __init__(self, decisions: DecisionsPairs, model: PairModel):
        self.decisions, self.model = decisions, model

    @property
    def version(self) -> str:
        return f"{self.model.version}+{PROMPT_VERSION}"

    def extract(self, text: str, mentions: dict, policy_types: list[tuple[str, str, str]], limits: Limits = Limits()) -> tuple[list[Evidence], list[Relation]]:
        codes = {c for c, _, _ in policy_types}
        pairs = candidate_pairs(text, mentions)
        if not pairs:
            return [], []
        P = positions(text, mentions)
        X = structure_features(text, P, pairs)
        if len(pairs) > MAX_PAIRS:                       # 같은 문장·제목 쌍 → 가까운 쌍 순
            keep = sorted(sorted(range(len(pairs)), key=lambda i: (-X[i, 0], -X[i, 2], X[i, 6], i))[:MAX_PAIRS])
            pairs, X = [pairs[i] for i in keep], X[keep]
        answers = self.decisions.ask(text, pairs)
        X = np.column_stack([X, decision_features(pairs, {(a, b): d for (a, b, _), d in zip(pairs, answers)})])
        p, types = self.model.predict(X)
        sents = sentence_spans(text)
        starts = [s for s, _ in sents]
        found: dict = {}
        for (a, b, _), pr, t in zip(pairs, p, types):
            if pr < self.model.threshold:
                continue
            decoded = decode(CLASSES[t])
            code = code_for(decoded[0], codes)
            if code is None:
                continue
            src, tgt = (a, b) if decoded[1] else (b, a)
            found[(src, code, tgt)] = {"evidence": _evidence_spans(text, P[a], P[b], sents, starts), "confidence": float(pr), "method": "pairs"}
        return finalize(text, found, limits)
