"""학생 모델의 특징과 데이터셋. 후보(키)마다 한 줄. 특징은 노트 안 모양·위치·구조·말뭉치 통계·교사 라벨 사전(leave-one-note-out)이다."""

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.services.concepts.config import STOPWORDS
from app.services.concepts.extract import concept_key
from app.services.concepts.gold import loose
from app.services.concepts.markdown import BOLD, HEADING, TABLE, mask, nfc, regions
from experiments.distill.candidates import generate
from experiments.llm_only.baseline import PROMPT_VERSION, SYSTEM, ground, parse

ROOT = Path(__file__).resolve().parents[2]
EN_STOP = frozenset("the a an of to for is are was were be been in on at by as and or not with from this that these those it its into over under between "
                    "than then so if when where which who whom what how why can could should would may might will do does did has have had but also such both each "
                    "their there here we our you your they them he she his her one two three more most less very only just about after before during".split())
TAGS = ("NNG", "NNP", "SL", "SH", "SN", "XR", "NR", "XSN")
FEATURES = (["n_tokens", "n_chars", "n_words", "hangul", "latin", "digit", "symbol", "upper_ratio", "first_upper", "all_caps", "hyphen", "paren"]
            + [f"first_{t}" for t in TAGS] + [f"last_{t}" for t in TAGS]
            + ["en_stop_first", "en_stop_last", "stop_frac", "kr_stop_first", "kr_stop_last", "ends_ko_verbish"]
            + ["tf", "log_tf", "first_pos", "n_heading", "n_bold", "n_table", "in_title", "standalone", "covered_frac", "doc_len_log", "sent_start_frac", "n_lines"]
            + ["log_df", "idf", "contained_in", "contains", "longer_tf", "old_score"]
            + ["paren_after", "in_paren", "colon_after", "quote_before", "bold_marker", "def_ko", "called_cue", "mid_cap", "acronym_pair", "tf_rel", "tf_rank", "in_list", "dash_after"]
            + ["lex_seen", "lex_rate", "first_seen", "first_rate", "last_seen", "last_rate"])


def teacher_concepts(text: str, effort: str = "medium"):
    """캐시된 LLM 개념 목록 → 원문에 있는 개념 [(대표 표기, loose 표기 집합)]. 캐시에 없으면 None (여기서는 LLM 을 부르지 않는다)."""
    from app.main import role_llm_config
    from app.services.llm import make_llm
    llm = make_llm(role_llm_config("relation", effort))
    for body in (nfc(text), text):
        key = hashlib.sha256(json.dumps([llm.provider, llm.model, effort, PROMPT_VERSION, SYSTEM, body], ensure_ascii=False).encode()).hexdigest()
        p = ROOT / "data/private/cache/llm-only" / f"{key}.json"
        if p.exists():
            grounded, _ = ground(parse(json.loads(p.read_text(encoding="utf-8"))), nfc(text), loose)
            return grounded
    return None


def teacher_forms(text: str, effort: str = "medium") -> frozenset | None:
    """교사 개념의 loose 표기 집합 (없으면 None)."""
    g = teacher_concepts(text, effort)
    return None if g is None else (frozenset().union(*[k for _, k in g]) if g else frozenset())


_CALLED = re.compile(r"(called|known as|named|so-called|termed|referred to as|stands for|denoted|dubbed|introduce[sd]?|propose[sd]?|define[sd]?)\s+(a |an |the )?[\"'“‘`*]*$", re.I)
_DEF_KO = re.compile(r"^\s*(은|는|이란|란|이라는|라는|이라고|라고|을 말|를 말|이라 함|라 함)")
_ACRO = re.compile(r"\(([A-Z][A-Za-z0-9\-]{1,9})\)")


def context_feats(text: str, spans, surf: str, tf_rel: float, tf_rank: float):
    """언급 앞뒤 글에서 읽는 단서(정의·별칭·강조). 언어를 가리지 않는 것만 쓴다. 앞 8개 언급의 비율."""
    m = spans[:8]
    n = len(m)
    c = defaultdict(int)
    for s, e in m:
        before, after = text[max(0, s - 24):s], text[e:e + 24]
        a = after.lstrip(" ")
        c["paren_after"] += a.startswith(("(", "（"))
        c["in_paren"] += before.rstrip(" ").endswith(("(", "（"))
        c["colon_after"] += a.startswith((":", "："))
        c["quote_before"] += before[-1:] in "\"'“‘`"
        c["bold_marker"] += before.endswith(("**", "__"))
        c["def_ko"] += bool(_DEF_KO.match(after))
        c["called"] += bool(_CALLED.search(before)) or bool(re.match(r"\s*(이?라고 (한다|부른다|부르|합니다|불리))", after))
        sent_start = s == 0 or text[max(0, s - 3):s].strip(" ") in ("", ".", "?", "!", "#", "-", "*", ">") or text[s - 1] == "\n"
        c["mid_cap"] += surf[:1].isupper() and not sent_start
        c["acro"] += bool(_ACRO.match(a)) or bool(re.search(r"\([^()]{2,40}\)\s*$", before) and surf.isupper())
        c["dash_after"] += a.startswith(("-", "–", "—", "→"))
    first = text[max(0, m[0][0] - 3):m[0][0]]
    in_list = sum(1 for s, _ in m if re.match(r"\s*([-*•]|\d+[.)])\s", text[text.rfind("\n", 0, s) + 1:s + 1]))
    return [c["paren_after"] / n, c["in_paren"] / n, c["colon_after"] / n, c["quote_before"] / n, c["bold_marker"] / n, c["def_ko"] / n, c["called"] / n,
            c["mid_cap"] / n, c["acro"] / n, tf_rel, tf_rank, in_list / n, c["dash_after"] / n]


def _words(key: str):
    return [w for w in re.split(r"[\s\-_/+]+", key) if w]


def note_rows(ex, text: str, df: dict[str, int], n_docs: int, old_scores: dict[str, float]):
    """한 노트의 후보 → (원문, [dict(key, surface, spans, x(list), words)]). 사전(lex) 특징은 나중에 채운다."""
    text, cands = generate(ex, text)
    masked = mask(text)
    regs = regions(masked)
    first_line_end = text.find("\n") if "\n" in text else len(text)
    title = loose(text[:first_line_end])
    keys = list(cands)
    # 언급 위치 → 더 긴 후보에 덮이는지
    spans = sorted((s, e, k) for k, c in cands.items() for s, e in c.spans)
    tf = {k: len(c.spans) for k, c in cands.items()}
    starts = [s for s, _, _ in spans]
    max_tf = max(tf.values()) if tf else 1
    order = sorted(tf.values())
    import bisect as _b
    import bisect
    covered = defaultdict(int)
    for s, e, k in spans:
        i = bisect.bisect_right(starts, s)
        j = i - 1
        while j >= 0 and spans[j][0] >= s - 60:
            ss, ee, kk = spans[j]
            if kk != k and ss <= s and ee >= e and (ee - ss) > (e - s) and tf[kk] >= 2:
                covered[k] += 1
                break
            j -= 1
    lower_keys = [k for k in keys]
    rows = []
    n = max(1, len(text))
    for k in keys:
        c = cands[k]
        surf = c.surface
        words = _words(k)
        n_head = n_bold = n_tab = 0
        for s, e in c.spans:
            for rs, re_, kind in regs:
                if rs <= s and e <= re_:
                    n_head += kind == HEADING
                    n_bold += kind == BOLD
                    n_tab += kind == TABLE
        longer = [o for o in lower_keys if o != k and len(o) > len(k) and k in o]
        shorter = [o for o in lower_keys if o != k and len(o) < len(k) and o in k]
        sent_start = sum(1 for s, _ in c.spans if s == 0 or text[max(0, s - 2):s].strip() in ("", ".", "?", "!", "#", "-", "*") or text[s - 1] == "\n")
        df_k = df.get(k, 0)
        x = [c.n_tokens, len(surf), len(surf.split()), bool(re.search(r"[가-힣]", surf)), bool(re.search(r"[A-Za-z]", surf)), bool(re.search(r"\d", surf)),
             bool(re.search(r"[^\w\s\-/+.가-힣]", surf)), sum(ch.isupper() for ch in surf) / max(1, len(surf)), surf[:1].isupper(), surf.isupper() and len(surf) > 1,
             "-" in surf, "(" in surf]
        x += [c.first_tag == t for t in TAGS] + [c.last_tag == t for t in TAGS]
        x += [words[0] in EN_STOP if words else 0, words[-1] in EN_STOP if words else 0, sum(w in EN_STOP or w in STOPWORDS for w in words) / max(1, len(words)),
              (words[0] in STOPWORDS) if words else 0, (words[-1] in STOPWORDS) if words else 0, surf[-1:] in "다요음함됨기"]
        x += [tf[k], math.log1p(tf[k]), c.spans[0][0] / n, n_head, n_bold, n_tab, bool(title and loose(surf) and loose(surf) in title), tf[k] - covered[k],
              covered[k] / tf[k], math.log(len(text) + 1), sent_start / tf[k], text.count("\n")]
        x += [math.log1p(df_k), math.log1p(n_docs / (1 + df_k)), len(longer), len(shorter), sum(tf[o] for o in longer), old_scores.get(k, 0.0)]
        x += context_feats(text, c.spans, surf, tf[k] / max_tf, _b.bisect_left(order, tf[k]) / max(1, len(order)))
        rows.append(dict(key=k, surface=surf, spans=c.spans, x=x, words=words))
    return text, rows


def old_scores(ex, note_id: str, text: str) -> dict[str, float]:
    """기존 추출기(임계값 0)의 키별 점수 — 학생의 한 특징일 뿐이다."""
    import dataclasses
    cfg = ex.config
    ex.config = dataclasses.replace(cfg, threshold=0.0)
    try:
        return {m.key: m.score for m in ex.extract(note_id, text)}
    finally:
        ex.config = cfg


class Lexicon:
    """교사 라벨에서 센 사전: 키·첫 낱말·끝 낱말이 개념으로 뽑힌 비율. 학습 줄은 자기 노트의 몫을 빼고(leave-one-note-out) 쓴다."""

    def __init__(self):
        self.k, self.f, self.l = {}, {}, {}      # key → [positives, rows seen]
        self.prior = 0.05

    def add(self, rows, labels):
        for r, y in zip(rows, labels):
            for table, k in ((self.k, r["key"]), (self.f, r["words"][0] if r["words"] else ""), (self.l, r["words"][-1] if r["words"] else "")):
                cell = table.setdefault(k, [0, 0])
                cell[0] += int(y)
                cell[1] += 1

    def feats(self, r, y_own=None):
        out = []
        for table, k in ((self.k, r["key"]), (self.f, r["words"][0] if r["words"] else ""), (self.l, r["words"][-1] if r["words"] else "")):
            pos, seen = table.get(k, (0, 0))
            if y_own is not None and seen:                          # 자기 노트의 몫 빼기
                pos -= int(y_own)
                seen -= 1
            out += [math.log1p(seen), (pos + 2 * self.prior) / (seen + 2)]
        return out
