"""높은 재현율 후보 생성: 토큰 n-gram. (실험용; 학생 모델이 후보마다 개념인지 고른다.)

기존 추출기(`ConceptExtractor.candidates`)는 영어 문장에서 명사구를 못 만들고(모든 영어 낱말을 명사로 읽는다) 낱말 사이 조각을 놓친다.
여기서는 코드·수식·주소를 가린 뒤(추출기와 같은 방식) 형태소 토큰의 연속 구간을 모두 후보로 낸다:
  - 처음과 끝 토큰이 명사류(NNG·NNP·SL·SH·SN·XR·NR)이고, 사이에 올 수 있는 것은 명사류·접사·연결 기호(- _ / +)뿐
  - 토큰 사이 간격은 없거나 공백 하나, 한 구간은 6토큰·60자까지, 줄·문장부호·조사·서술어에서 끊는다
후보는 정규화 키(concept_key)로 묶고 언급 위치를 모은다. 판단(개념인가)은 하지 않는다.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.services.concepts.extract import CONNECTORS, DETACHED_PARTICLES, INNER_TAGS, NOUN_TAGS, _after_symbol, concept_key
from app.services.concepts.markdown import mask, nfc, strip_inline_marks

START_TAGS = frozenset({"NNG", "NNP", "SL", "SH", "SN", "XR", "NR"})
END_TAGS = frozenset({"NNG", "NNP", "SL", "SH", "SN", "XR", "NR", "XSN"})
MIDDLE_TAGS = frozenset(INNER_TAGS | {"XR", "NR", "XSN", "XPN"})
MAX_TOKENS, MAX_CHARS = 6, 60


@dataclass
class Cand:
    key: str
    surface: str                                   # 가장 많이 쓰인 표기
    spans: list = field(default_factory=list)      # [(start, end)] 원문 위치 (코드 포인트)
    first_tag: str = ""
    last_tag: str = ""
    n_tokens: int = 1


def tokens(ex, text: str):
    """(NFC 원문, 토큰 목록). 토큰 = (form, tag, start, end), 가린 구간에 걸치면 tag 를 'HID' 로 한다."""
    text = nfc(text)
    masked = mask(text)
    hidden = [m == " " and c not in " \t\n" for m, c in zip(masked, text)]
    clean, pos = strip_inline_marks("".join("X" if h else m for h, m in zip(hidden, masked)))
    raw = []
    for t in ex._tokens(clean, pos):
        tag = "HID" if any(hidden[t.start:t.start + t.len]) else t.tag
        if t.tag.startswith(("NN", "XPN")) and t.form in DETACHED_PARTICLES and _after_symbol(text, t.start):
            tag = "BRK"                              # `x`가 처럼 가린 코드 뒤의 조사
        raw.append([t.form, tag, t.start, t.start + t.len])
    # 형태소 분석기가 영어 낱말 하나를 둘로 쪼개거나 ("work w" + "ell") 끝의 문장부호를 낱말에 붙이는 ("well.") 일이 있다
    merged = []
    for tok in raw:
        if merged and tok[1] == "SL" and merged[-1][1] == "SL" and merged[-1][3] == tok[2]:
            merged[-1][3] = tok[3]
            merged[-1][0] = text[merged[-1][2]:tok[3]]
        else:
            merged.append(tok)
    out = []
    for form, tag, s, e in merged:
        if tag in ("SL", "SH", "SN"):
            m = re.search(r"[.,;:!?]+$", text[s:e])
            if m and e - len(m.group()) > s:
                e -= len(m.group())
                out.append((text[s:e], tag, s, e))
                out.append((".", "BRK", e, e + len(m.group())))
                continue
        out.append((form, tag, s, e))
    return text, out, masked


def segments(text: str, toks: list, masked: str):
    """끊김 없이 이어지는 토큰 묶음들. 끊는 것: 명사류가 아닌 토큰, 가린 구간, 줄바꿈이 낀 간격, 공백 둘 이상."""
    seg, prev_end = [], None
    for form, tag, s, e in toks:
        is_conn = tag == "SO" and form in CONNECTORS
        ok = tag in MIDDLE_TAGS or (is_conn and seg)
        if ok:
            gap = masked[prev_end:s] if prev_end is not None else ""
            if seg and gap not in ("", " "):
                yield seg
                seg = []
            seg.append((form, tag, s, e))
            prev_end = e
        else:
            if seg:
                yield seg
            seg, prev_end = [], None
    if seg:
        yield seg


def generate(ex, text: str) -> tuple[str, dict[str, Cand]]:
    """(NFC 원문, 키 → 후보). 같은 키의 언급 위치는 모두 모은다."""
    text, toks, masked = tokens(ex, text)
    out: dict[str, Cand] = {}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seg in segments(text, toks, masked):
        n = len(seg)
        for i in range(n):
            if seg[i][1] not in START_TAGS:
                continue
            for j in range(i, min(n, i + MAX_TOKENS)):
                if seg[j][1] not in END_TAGS:
                    continue
                s, e = seg[i][2], seg[j][3]
                if e - s > MAX_CHARS:
                    break
                surface = text[s:e]
                key = concept_key(surface)
                if not key or not re.search(r"[A-Za-z가-힣]", key) or not ex._keep(key):
                    continue
                c = out.get(key)
                if c is None:
                    c = out[key] = Cand(key, surface, [], seg[i][1], seg[j][1], j - i + 1)
                c.spans.append((s, e))
                counts[key][surface] += 1
    for key, c in out.items():
        c.surface = sorted(counts[key], key=lambda s: (-counts[key][s], s))[0]
    return text, out
