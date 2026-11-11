"""계산용 사실: 노트 문장에서 구조화한 전제 (R1 구현 적용 · R2 비용 합성의 입력).

사실 하나는 원문 근거 하나에 묶인다. 근거 밖 지식으로 사실을 만들지 않는다 (모델 상식으로 빠진 전제를 채우지 않는다).
역할(role)·연산(op)은 영문 소문자·밑줄 key 다. 자주 나오는 것은 아래 ROLES·OPS 의 key 를 쓰고(노트마다 다르게 쓴 표기
"키 감소", "decrease-key", "decreaseKey" 가 같은 연산이 된다), 거기 없는 것(DB 인덱스 조회, 공간 인덱스 질의, 캐시 적중 …)은
새 key 를 만들어 쓴다. 새 key 는 roleLabel·opLabel 에 한국어 이름을 함께 적어 설명에 쓴다. 노트 사이 같은 key 를 쓰게 하는 일은
추출기가 이미 쓰인 key 목록(knownRoles·knownOps)을 보고 맞춘다.
주체(알고리즘·자료 구조)는 subjectKey 로 잇는다: 영문 소문자·밑줄의 정규 이름 (dijkstra, indexed_binary_min_heap).

종류 (kind)
  needs_op      알고리즘 subject 가 역할 role 에서 연산 op 를 쓴다. expr = 최대 횟수 (없으면 횟수 모름)
  uses          알고리즘 subject 가 역할 role 을 쓴다 (어떤 연산인지는 말하지 않음)
  size_bound    알고리즘 subject 에서 역할 role 의 크기 n ≤ expr
  extra_cost    알고리즘 subject 에서 역할 연산 밖의 비용 expr
  implements    자료 구조 subject 가 역할 role 의 구현이다
  op_cost       자료 구조 subject 가 연산 op 를 제공한다. expr = 크기 n 에 대한 비용 (없으면 비용 모름). case = worst | amortized | average
                negated = true 면 그 연산을 제공하지 않는다 (또는 그 비용으로 하지 못한다)
  requires      subject 가 성립하려면 condition 이 필요하다 (비음수 가중치, 비교 가능한 키)
  stated_result 노트가 이미 쓴 결론: subject(알고리즘)를 other(자료 구조)로 구현하면 expr (expr 없으면 '쓸 수 있다' 만)
"""

import hashlib
import re
from dataclasses import dataclass, field

KINDS = ("needs_op", "uses", "size_bound", "extra_cost", "implements", "op_cost", "requires", "stated_result")
CASES = ("worst", "amortized", "average")

KEY_RE = r"[a-z][a-z0-9_]{0,39}"

# 자주 나오는 역할(추상 자료형) → 한국어 이름
ROLES = {
    "priority_queue": "우선순위 큐", "queue": "큐", "stack": "스택", "deque": "덱", "dictionary": "사전(해시 테이블)",
    "set": "집합", "disjoint_set": "서로소 집합", "ordered_set": "순서 있는 집합", "list": "리스트",
    "array": "배열", "graph": "그래프", "index": "인덱스", "cache": "캐시",
    "sorted_sequence": "정렬된 열", "tree": "트리",
}
# 자주 나오는 연산 → 한국어 이름 (순서는 설명에서 연산을 늘어놓는 순서)
OPS = {
    "insert": "삽입", "extract_min": "최솟값 꺼내기", "extract_max": "최댓값 꺼내기", "find_min": "최솟값 보기",
    "find_max": "최댓값 보기", "decrease_key": "키 줄이기", "increase_key": "키 늘리기", "delete": "삭제",
    "search": "찾기", "update": "갱신", "push": "push", "pop": "pop", "peek": "맨 위 보기",
    "enqueue": "넣기(enqueue)", "dequeue": "빼기(dequeue)", "union": "합치기(union)", "find": "대표 찾기(find)",
    "make_set": "집합 만들기", "build": "만들기(build)", "merge": "병합", "access": "접근", "successor": "다음 원소 찾기",
    "adjacent": "인접 정점 나열",
    "lookup": "조회", "probe": "탐색(probe)", "range_query": "범위 질의", "scan": "훑기",
    "evict": "내보내기(evict)",
}

# 같은 연산·역할을 노트마다 달리 부르는 말 → 대표 key. LLM 이 고른 key 를 이것으로 맞춘다 (노트 사이 key 가 같아야 이어진다)
OP_SYNONYMS = {
    "lookup": "search", "point_lookup": "search", "probe": "search", "get": "search", "contains": "search",
    "find_key": "search", "search_key": "search", "query": "search",
    "region_query": "range_query", "radius_query": "range_query", "range_search": "range_query",
    "neighbor_query": "range_query", "epsilon_query": "range_query", "radius_search": "range_query",
    "put": "insert", "add": "insert", "remove": "delete", "erase": "delete",
    "pop_min": "extract_min", "delete_min": "extract_min", "remove_min": "extract_min",
    "pop_max": "extract_max", "delete_max": "extract_max", "remove_max": "extract_max",
    "get_min": "find_min", "peek_min": "find_min", "get_max": "find_max", "peek_max": "find_max",
}
ROLE_SYNONYMS = {"hash_table": "dictionary", "hash_map": "dictionary", "map": "dictionary", "hashmap": "dictionary",
                 "hashtable": "dictionary", "heap": "priority_queue", "pq": "priority_queue"}


def canon_op(key: str | None) -> str | None:
    return OP_SYNONYMS.get(key, key) if key else key


def canon_role(key: str | None) -> str | None:
    return ROLE_SYNONYMS.get(key, key) if key else key
SIZE_VAR = "n"          # op_cost 식의 크기 변수 이름 (추출 때 이것으로 맞춘다)


def subject_key(text: str) -> str:
    """정규 이름 비교용 key: 소문자, 공백·하이픈 → 밑줄."""
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _squash(text: str | None) -> str | None:
    return re.sub(r"\s+", "", text) if text else None          # 식: 공백을 무시


def _words(text: str | None) -> str | None:
    return " ".join(text.split()) if text else None            # 조건 문장: 연속 공백을 하나로


def _digest(*parts) -> str:
    """값이 None 인 칸과 빈 문자열을 구분하고 칸 경계를 구분하는 sha256 (hex 64자)."""
    raw = "\x1f".join("\x00" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def facts_hash(facts: list["Fact"]) -> str:
    """한 노트의 사실 묶음의 뜻 해시: 사실의 content_hash 를 정렬해 합친다 (순서·id·위치와 무관, 같은 사실이 둘이면 둘로 센다).
    노트를 고쳐도 이 값이 같으면 사실이 같다 — 그 사실로 만든 결론은 그대로 유효하다."""
    return _digest("facts-set-1", *sorted(f.content_hash() for f in facts))


@dataclass(frozen=True)
class FactEvidence:
    note_id: str
    note_version: int
    body_sha256: str
    start: int          # UTF-16
    end: int
    quote: str


@dataclass(frozen=True)
class Fact:
    fact_id: str
    kind: str
    subject: str                    # 노트에 쓴 이름
    subject_key: str
    evidence: FactEvidence
    role: str | None = None
    op: str | None = None
    role_label: str | None = None   # 새 key 의 한국어 이름 (ROLES 에 있으면 필요 없다)
    op_label: str | None = None
    expr: str | None = None
    case: str | None = None
    negated: bool = False
    condition: str | None = None
    other: str | None = None        # stated_result 의 자료 구조 이름
    other_key: str | None = None
    origin: str = "extracted"       # extracted | manual (사람이 적은 전제는 자동 추출 성능과 따로 잰다)

    @property
    def note(self) -> str:
        return self.evidence.note_id

    def content_hash(self) -> str:
        """사실의 뜻만 해시한다 (종류·주체·역할·연산·식·경우·부정·조건). 사실 id·노트에 쓴 이름·역할/연산의 한국어 이름·출처·근거 위치는
        넣지 않는다: 노트의 다른 곳을 고치거나 다시 추출해 id 와 위치가 바뀌어도 뜻이 같으면 같은 값이다. 식의 공백은 무시한다."""
        return _digest("fact-content-1", self.kind, self.subject_key, self.role, self.op, _squash(self.expr), self.case, self.negated,
                       _words(self.condition), self.other_key)

    def text(self) -> str:
        """사실을 한 문장으로 (soundness 판정·설명에 쓴다)."""
        role = self.role_label or ROLES.get(self.role or "", self.role or "")
        op = self.op_label or OPS.get(self.op or "", self.op or "")
        no = "하지 않는다" if self.negated else ""
        if self.kind == "needs_op":
            return f"{self.subject} 는 {role} 의 {op} 연산을 " + (f"최대 {self.expr} 번 한다" if self.expr else "쓴다")
        if self.kind == "uses":
            return f"{self.subject} 는 {role} 을 쓴다"
        if self.kind == "size_bound":
            return f"{self.subject} 에서 {role} 의 크기 n 은 최대 {self.expr} 이다"
        if self.kind == "extra_cost":
            return f"{self.subject} 의 {role or '역할'} 연산 밖 비용은 O({self.expr}) 이다"
        if self.kind == "implements":
            return f"{self.subject} 는 {role} 의 구현이다" if not self.negated else f"{self.subject} 는 {role} 의 구현이 아니다"
        if self.kind == "op_cost":
            if self.negated:
                return f"{self.subject} 는 {op} 연산을 {('O(' + self.expr + ') 에 ') if self.expr else ''}제공{no}"
            case = {"amortized": "상각 ", "average": "평균 ", "worst": "최악 "}.get(self.case or "", "")
            return f"{self.subject} 는 {op} 연산을 제공한다" + (f" ({case}O({self.expr}), n 은 크기)" if self.expr else "")
        if self.kind == "requires":
            return f"{self.subject} 는 '{self.condition}' 이 필요하다"
        if self.kind == "stated_result":
            return f"{self.subject} 를 {self.other} 로 구현하면 " + (f"O({self.expr}) 이다" if self.expr else "쓸 수 있다")
        return f"{self.subject}: {self.kind}"


@dataclass
class FactProblem:
    fact_id: str
    reason: str


def check_fact(f: Fact) -> str | None:
    """사실 자체의 모양 검사. 문제가 있으면 이유 (그 사실은 도출에 쓰지 않는다)."""
    need = {"needs_op": ("role", "op"), "uses": ("role",), "size_bound": ("role", "expr"), "extra_cost": ("expr",),
            "implements": ("role",), "op_cost": ("op",), "requires": ("condition",), "stated_result": ("other_key",)}
    if f.kind not in KINDS:
        return "unknown_kind"
    for attr in need[f.kind]:
        if not getattr(f, attr):
            return f"missing_{attr}"
    if f.role and not re.fullmatch(KEY_RE, f.role):
        return "invalid_role"
    if f.op and not re.fullmatch(KEY_RE, f.op):
        return "invalid_op"
    if f.case and f.case not in CASES:
        return "unknown_case"
    if not re.fullmatch(r"[a-z0-9_]{1,80}", f.subject_key) or (f.other_key and not re.fullmatch(r"[a-z0-9_]{1,80}", f.other_key)):
        return "invalid_key"
    return None


@dataclass
class FactSet:
    """도출 입력: 쓸 수 있는 사실과 쓰지 않은 사실(이유)."""
    facts: list[Fact] = field(default_factory=list)
    problems: list[FactProblem] = field(default_factory=list)

    @classmethod
    def of(cls, facts: list[Fact]) -> "FactSet":
        out = cls()
        for f in facts:
            why = check_fact(f)
            if why:
                out.problems.append(FactProblem(f.fact_id, why))
            else:
                out.facts.append(f)
        return out
