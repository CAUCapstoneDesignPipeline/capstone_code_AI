"""Big-O 상한 식의 작은 대수 (R2 비용 합성).

노트에 적힌 식("O(log n)", "V+E", "|E|", "n²", "E α(V)")을 읽어 계수를 버린 다항식(단항식들의 합)으로 다룬다.
  - 곱·합·거듭제곱, 변수 대입 (n := V, 큐 크기 n ≤ V 를 상한에 넣기)
  - 빼기는 버린다: a − b ≤ a 라서 상한으로는 맞다 ("V−1" → V)
  - 나눗셈·지수 함수(2^n)·음수 지수는 읽지 않는다 (ParseError → 그 도출은 보류)
  - 지배되는 항을 지운다: V ≤ V log V, log V ≤ V, 상수 ≤ 무엇이든 (변수는 2 이상이라고 본다)
식이 같은지는 정리한 다항식이 같은지로 본다: O(V log V + E log V) = O((V+E) log V).
"""

import re
from dataclasses import dataclass

# 원자: ("var", "V") 또는 ("log", 인자 다항식) / ("alpha", 인자 다항식). 느린 함수(log·α)는 어떤 변수보다 느리게 자란다
Atom = tuple
Mono = tuple            # ((원자, 지수), …) 원자 순으로 정렬. () 은 상수
Poly = frozenset        # 단항식들


class ParseError(ValueError):
    pass


_SLOW = {"log": "log", "lg": "log", "ln": "log", "alpha": "alpha", "α": "alpha"}
_VAR_ORDER = {"V": 0, "E": 1, "n": 2, "m": 3, "k": 4}


def _norm_text(s: str) -> str:
    s = s.strip().replace("$", "")
    s = re.sub(r"\\(log|lg|ln|alpha|cdot|times|left|right|,|;|!)", lambda m: {
        "cdot": "*", "times": "*", "alpha": "α", "left": "", "right": "", ",": " ", ";": " ", "!": " "}.get(
        m.group(1), m.group(1)), s)
    s = s.replace("\\{", "(").replace("\\}", ")").replace("{", "(").replace("}", ")")
    s = s.replace("·", "*").replace("×", "*").replace("⋅", "*").replace("−", "-").replace("–", "-")
    s = s.replace("²", "^2").replace("³", "^3")
    s = re.sub(r"\|\s*([A-Za-z])\s*\|", r"\1", s)                  # |V| → V
    m = re.fullmatch(r"\s*(?:O|Θ|θ|\\Theta|\\mathcal\s*O|\\mathcal\(O\))\s*\((.*)\)\s*", s)
    if m:
        s = m.group(1)
    if re.match(r"\s*(?:Ω|\\Omega|o|ω)\s*\(", s):
        raise ParseError("하한·작은 o 는 상한이 아니다")
    return s


_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-zα][A-Za-z0-9]*)|(\^|\*|\+|-|\(|\)|_))")


def _tokens(s: str) -> list[str]:
    out, pos = [], 0
    s = s.rstrip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ParseError(f"읽지 못한 글자: {s[pos:pos + 5]!r}")
        tok = m.group(1) or m.group(2) or m.group(3)
        if m.group(2) and m.group(2) not in _SLOW and m.group(2)[:3] in ("log",) and len(m.group(2)) > 3:
            out += ["log", m.group(2)[3:]]                       # logn → log n
        else:
            out.append(tok)
        pos = m.end()
    return out


def _mono_mul(a: Mono, b: Mono) -> Mono:
    d = dict(a)
    for atom, e in b:
        d[atom] = d.get(atom, 0) + e
    return tuple(sorted(((k, v) for k, v in d.items() if v), key=_atom_key))


def mul(p: Poly, q: Poly) -> Poly:
    return frozenset(_mono_mul(a, b) for a in p for b in q)


def add(*ps: Poly) -> Poly:
    return frozenset().union(*ps)


def power(p: Poly, k: float) -> Poly:
    if len(p) == 1:
        (m,) = p
        return frozenset({tuple((a, e * k) for a, e in m)})
    if k != int(k) or k < 0:
        raise ParseError("합의 분수 거듭제곱")
    out: Poly = frozenset({()})
    for _ in range(int(k)):
        out = mul(out, p)
    return out


def var(name: str) -> Poly:
    return frozenset({((("var", name), 1),)})


CONST: Poly = frozenset({()})


def slow(kind: str, arg: Poly) -> Poly:
    """log(인자) / α(인자). log(상수) = 상수, log(a·b) = log a + log b, log(a^k) = log a (계수 버림)."""
    arg = simplify(arg)
    if arg == CONST:
        return CONST
    if kind == "log" and len(arg) == 1:
        (m,) = arg
        parts = []
        for atom, _ in m:
            if atom[0] == "var":
                parts.append(frozenset({((("log", frozenset({((atom, 1),)})), 1),)}))
            else:                                               # log(log n) 은 log n 보다 작다 → log n 으로 올려 둔다
                parts.append(slow("log", atom[1]))
        return add(*parts)
    return frozenset({(((kind, arg), 1),)})


class _Parser:
    def __init__(self, toks: list[str]):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, tok=None):
        cur = self.peek()
        if cur is None or (tok is not None and cur != tok):
            raise ParseError(f"기대: {tok}, 받음: {cur}")
        self.i += 1
        return cur

    def expr(self) -> Poly:
        terms = [self.term()]
        while self.peek() in ("+", "-"):
            op = self.eat()
            t = self.term()
            if op == "+":
                terms.append(t)                                  # 빼는 항은 버린다 (상한)
        return add(*terms)

    def term(self) -> Poly:
        p = self.factor()
        while True:
            nxt = self.peek()
            if nxt == "*":
                self.eat()
                p = mul(p, self.factor())
            elif nxt is not None and nxt not in ("+", "-", ")", "^", "_"):   # 붙여 쓴 곱 (E log V)
                p = mul(p, self.factor())
            else:
                return p

    def factor(self) -> Poly:
        base = self.base()
        if self.peek() == "^":
            self.eat()
            k = self.eat()
            if not re.fullmatch(r"\d+(?:\.\d+)?", k):
                raise ParseError("변수 지수 (지수 함수)")
            base = power(base, float(k) if "." in k else int(k))
        return base

    def base(self) -> Poly:
        tok = self.eat()
        if tok == "(":
            p = self.expr()
            self.eat(")")
            return p
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            if self.peek() == "^":
                raise ParseError("지수 함수")
            return CONST
        if tok in _SLOW:
            kind = _SLOW[tok]
            if self.peek() == "_":                              # log_2 n
                self.eat()
                self.eat()
            k = 1
            if self.peek() == "^":                              # log^2 n
                self.eat()
                k = int(self.eat())
            arg = self.factor() if self.peek() != "(" else (self.eat("("), self.expr(), self.eat(")"))[1]
            return power(slow(kind, arg), k)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tok):
            if len(tok) > 1 and not tok[1:].isdigit() and tok not in ("sqrt",):
                if tok.isalpha() and tok.isupper() and len(tok) <= 3:   # VE → V·E
                    return _join(tok)
                raise ParseError(f"모르는 이름: {tok}")
            if tok == "sqrt":
                self.eat("(")
                p = self.expr()
                self.eat(")")
                return power(p, 0.5)
            return var(tok)
        raise ParseError(f"모르는 토큰: {tok}")


def _join(s: str) -> Poly:
    out = CONST
    for c in s:
        out = mul(out, var(c))
    return out


def parse(text: str) -> Poly:
    toks = _tokens(_norm_text(text))
    if not toks:
        raise ParseError("빈 식")
    p = _Parser(toks)
    out = p.expr()
    if p.peek() is not None:
        raise ParseError(f"남은 토큰: {p.peek()}")
    return simplify(out)


def substitute(p: Poly, name: str, value: Poly) -> Poly:
    """변수 name 을 다항식 value 로 바꾼다 (느린 함수 인자 안까지)."""
    out = []
    for m in p:
        acc = CONST
        for atom, e in m:
            if atom == ("var", name):
                acc = mul(acc, power(value, e))
            elif atom[0] in ("log", "alpha"):
                acc = mul(acc, power(slow(atom[0], substitute(atom[1], name, value)), e))
            else:
                acc = mul(acc, frozenset({((atom, e),)}))
        out.append(acc)
    return simplify(add(*out))


def variables(p: Poly) -> set[str]:
    out = set()
    for m in p:
        for atom, _ in m:
            if atom[0] == "var":
                out.add(atom[1])
            else:
                out |= variables(atom[1])
    return out


def _dominated(a: Mono, b: Mono) -> bool:
    """a = O(b) 인가 (변수 ≥ 2). 변수 지수가 모두 b 이하이고, 느린 함수는 b 에도 있거나 b 의 어떤 변수 지수가 더 커서 덮인다."""
    if a == b:
        return False
    da, db = dict(a), dict(b)
    vars_a = {x: e for x, e in da.items() if x[0] == "var"}
    vars_b = {x: e for x, e in db.items() if x[0] == "var"}
    if any(e > vars_b.get(x, 0) for x, e in vars_a.items()):
        return False
    strictly_more = any(e > vars_a.get(x, 0) for x, e in vars_b.items())
    for x, e in da.items():
        if x[0] != "var" and e > db.get(x, 0) and not strictly_more:
            return False
    return True


def simplify(p: Poly) -> Poly:
    p = frozenset(p)
    keep = [m for m in p if not any(_dominated(m, o) for o in p)]
    return frozenset(keep) if keep else CONST


def equivalent(a: Poly, b: Poly) -> bool:
    return simplify(a) == simplify(b)


# ---- 출력 ----

def _atom_key(item) -> tuple:
    atom = item[0] if isinstance(item[0], tuple) else item
    if atom[0] == "var":
        return (0, _VAR_ORDER.get(atom[1], 9), atom[1])
    return (1 if atom[0] == "log" else 2, 0, show(atom[1]))


def _show_atom(atom: Atom, e) -> str:
    if atom[0] == "var":
        s = atom[1]
        return s if e == 1 else (f"√{s}" if e == 0.5 else f"{s}^{_num(e)}")
    inner = show(atom[1])
    name = "log" if atom[0] == "log" else "α"
    if atom[0] == "log" and len(atom[1]) == 1 and len(next(iter(atom[1]))) == 1 and next(iter(atom[1]))[0][1] == 1:
        core = f"log {inner}"
    else:
        core = f"{name}({inner})"
    return core if e == 1 else (f"log^{_num(e)} {inner}" if core.startswith("log ") else f"{core}^{_num(e)}")


def _num(e) -> str:
    return str(int(e)) if e == int(e) else str(e)


def _show_mono(m: Mono) -> str:
    if not m:
        return "1"
    items = sorted(m, key=_atom_key)
    vars_ = [_show_atom(a, e) for a, e in items if a[0] == "var"]
    rest = [_show_atom(a, e) for a, e in items if a[0] != "var"]
    return " ".join(x for x in ("·".join(vars_), *rest) if x)


def _mono_order(m: Mono) -> tuple:
    return tuple(_atom_key(it) + (-it[1],) for it in sorted(m, key=_atom_key))


def show(p: Poly) -> str:
    """사람이 읽는 식 (O 없이). 모든 항에 공통인 인수는 묶는다: V log V + E log V → (V + E) log V."""
    p = simplify(p)
    monos = sorted(p, key=_mono_order)
    if len(monos) > 1:
        common = dict(monos[0])
        for m in monos[1:]:
            d = dict(m)
            common = {a: min(e, d[a]) for a, e in common.items() if a in d}
        if common:
            g = tuple(sorted(common.items(), key=_atom_key))
            rest = []
            for m in monos:
                d = dict(m)
                for a, e in g:
                    d[a] -= e
                rest.append(tuple(sorted(((a, e) for a, e in d.items() if e), key=_atom_key)))
            return f"({' + '.join(_show_mono(m) for m in rest)}) {_show_mono(g)}"
    return " + ".join(_show_mono(m) for m in monos)


def big_o(p: Poly) -> str:
    return f"O({show(p)})"


@dataclass(frozen=True)
class Expr:
    """원문 식과 읽은 다항식."""
    text: str
    poly: Poly

    @classmethod
    def read(cls, text: str) -> "Expr":
        return cls(text, parse(text))
