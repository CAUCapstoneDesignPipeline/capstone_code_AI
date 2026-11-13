"""학생 개념 선택기 (`extract-student-1`): 토큰 n-gram 후보(ngrams.py) 가운데 개념인 것을 빠르게 고른다. LLM 을 부르지 않는다 (한 노트 약 0.1초).

후보마다 모양·위치·구조·문맥 단서 특징 (코퍼스 통계나 노트 내용에서 센 사전은 쓰지 않는다: 어떤 분야·어떤 사용자의 노트에도 같은 모델이 돌고,
모델 파일에 노트 속 낱말이 들어 있지 않다) → 그래디언트 부스팅 트리의 확률 → 임계값 → 겹침 정리 (더 긴 채택 후보 안에서만 나오는 짧은 후보는 뺀다).
트리는 `student-1.json` 에 있고 numpy 로 한꺼번에 계산한다. 같은 입력은 항상 같은 결과다.
학습: `experiments/distill/train_student.py` (LLM 이 뽑은 개념 목록을 교사로 한다). 측정과 한계: docs/eval/concept-student.md.
"""

import bisect
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import STOPWORDS
from .gold import loose
from .markdown import BOLD, HEADING, TABLE, Utf16Index, mask, regions
from .ngrams import generate
from .schema import Mention, Method, Span

EN_STOP = frozenset("the a an of to for is are was were be been in on at by as and or not with from this that these those it its into over under between "
                    "than then so if when where which who whom what how why can could should would may might will do does did has have had but also such both each "
                    "their there here we our you your they them he she his her one two three more most less very only just about after before during".split())
TAGS = ("NNG", "NNP", "SL", "SH", "SN", "XR", "NR", "XSN")
FEATURES = (["n_tokens", "n_chars", "n_words", "hangul", "latin", "digit", "symbol", "upper_ratio", "first_upper", "all_caps", "hyphen", "paren"]
            + [f"first_{t}" for t in TAGS] + [f"last_{t}" for t in TAGS]
            + ["en_stop_first", "en_stop_last", "stop_frac", "kr_stop_first", "kr_stop_last", "ends_ko_verbish"]
            + ["tf", "log_tf", "first_pos", "n_heading", "n_bold", "n_table", "in_title", "standalone", "covered_frac", "doc_len_log", "sent_start_frac", "n_lines"]
            + ["contained_in", "contains", "longer_tf"]
            + ["paren_after", "in_paren", "colon_after", "quote_before", "bold_marker", "def_ko", "called_cue", "mid_cap", "acronym_pair", "tf_rel", "tf_rank",
               "in_list", "dash_after"])
MAX_CONCEPTS, MAX_MENTIONS = 200, 20
_CALLED = re.compile(r"(called|known as|named|so-called|termed|referred to as|stands for|denoted|dubbed|introduce[sd]?|propose[sd]?|define[sd]?)\s+(a |an |the )?[\"'“‘`*]*$", re.I)
_DEF_KO = re.compile(r"^\s*(은|는|이란|란|이라는|라는|이라고|라고|을 말|를 말|이라 함|라 함)")
_ACRO = re.compile(r"\(([A-Z][A-Za-z0-9\-]{1,9})\)")


def _words(key: str):
    return [w for w in re.split(r"[\s\-_/+]+", key) if w]


def _context(text: str, spans, surf: str, tf_rel: float, tf_rank: float) -> list[float]:
    """언급 앞뒤 글에서 읽는 정의·별칭·강조 단서 (앞 8개 언급의 비율)."""
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
    in_list = sum(1 for s, _ in m if re.match(r"\s*([-*•]|\d+[.)])\s", text[text.rfind("\n", 0, s) + 1:s + 1]))
    return [c["paren_after"] / n, c["in_paren"] / n, c["colon_after"] / n, c["quote_before"] / n, c["bold_marker"] / n, c["def_ko"] / n, c["called"] / n,
            c["mid_cap"] / n, c["acro"] / n, tf_rel, tf_rank, in_list / n, c["dash_after"] / n]


def rows(extractor, text: str):
    """(NFC 원문, [dict(key, surface, spans, x)]). x 는 FEATURES 순서."""
    text, cands = generate(extractor, text)
    masked = mask(text)
    regs = regions(masked)
    first_line_end = text.find("\n") if "\n" in text else len(text)
    title = loose(text[:first_line_end])
    keys = list(cands)
    spans = sorted((s, e, k) for k, c in cands.items() for s, e in c.spans)
    tf = {k: len(c.spans) for k, c in cands.items()}
    starts = [s for s, _, _ in spans]
    max_tf = max(tf.values()) if tf else 1
    order = sorted(tf.values())
    covered = defaultdict(int)
    for s, e, k in spans:                                              # 언급이 더 긴 후보(두 번 이상 나온)에 덮이는가
        j = bisect.bisect_right(starts, s) - 1
        while j >= 0 and spans[j][0] >= s - 60:
            ss, ee, kk = spans[j]
            if kk != k and ss <= s and ee >= e and (ee - ss) > (e - s) and tf[kk] >= 2:
                covered[k] += 1
                break
            j -= 1
    out = []
    n_chars = max(1, len(text))
    n_lines = text.count("\n")
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
        longer = [o for o in keys if o != k and len(o) > len(k) and k in o]
        shorter = [o for o in keys if o != k and len(o) < len(k) and o in k]
        sent_start = sum(1 for s, _ in c.spans if s == 0 or text[max(0, s - 2):s].strip() in ("", ".", "?", "!", "#", "-", "*") or text[s - 1] == "\n")
        x = [c.n_tokens, len(surf), len(surf.split()), bool(re.search(r"[가-힣]", surf)), bool(re.search(r"[A-Za-z]", surf)), bool(re.search(r"\d", surf)),
             bool(re.search(r"[^\w\s\-/+.가-힣]", surf)), sum(ch.isupper() for ch in surf) / max(1, len(surf)), surf[:1].isupper(), surf.isupper() and len(surf) > 1,
             "-" in surf, "(" in surf]
        x += [c.first_tag == t for t in TAGS] + [c.last_tag == t for t in TAGS]
        x += [bool(words) and words[0] in EN_STOP, bool(words) and words[-1] in EN_STOP, sum(w in EN_STOP or w in STOPWORDS for w in words) / max(1, len(words)),
              bool(words) and words[0] in STOPWORDS, bool(words) and words[-1] in STOPWORDS, surf[-1:] in "다요음함됨기"]
        x += [tf[k], math.log1p(tf[k]), c.spans[0][0] / n_chars, n_head, n_bold, n_tab, bool(title and loose(surf) and loose(surf) in title), tf[k] - covered[k],
              covered[k] / tf[k], math.log(len(text) + 1), sent_start / tf[k], n_lines]
        x += [len(longer), len(shorter), sum(tf[o] for o in longer)]
        x += _context(text, c.spans, surf, tf[k] / max_tf, bisect.bisect_left(order, tf[k]) / max(1, len(order)))
        out.append(dict(key=k, surface=surf, spans=c.spans, x=x))
    return text, out


def suppress_nested(rows_, probs, threshold: float) -> list[int]:
    """임계값 이상인 후보 중, 모든 언급이 더 긴 채택 후보 안에 든 짧은 후보를 뺀 인덱스."""
    chosen = sorted((i for i, p in enumerate(probs) if p >= threshold), key=lambda i: (-len(rows_[i]["surface"]), -probs[i]))
    kept, kept_spans = [], []
    for i in chosen:
        if kept_spans and all(any(ks <= s and e <= ke and (ke - ks) > (e - s) for ks, ke in kept_spans) for s, e in rows_[i]["spans"]):
            continue
        kept.append(i)
        kept_spans += rows_[i]["spans"]
    return kept


class Student:
    def __init__(self, data: dict):
        if list(data["features"]) != FEATURES:
            raise ValueError("모델 파일의 특징이 코드와 다르다 — 다시 학습해야 한다")
        self.id, self.threshold, self.baseline = data["id"], float(data["threshold"]), float(data["baseline"])
        trees = data["trees"]
        width = max(len(t["feature"]) for t in trees)
        pad = lambda key, dtype, fill: np.array([t[key] + [fill] * (width - len(t[key])) for t in trees], dtype=dtype)
        self.feature, self.cut = pad("feature", np.int64, 0), pad("threshold", np.float64, 0.0)
        self.left, self.right, self.value = pad("left", np.int64, -1), pad("right", np.int64, -1), pad("value", np.float64, 0.0)
        self.depth = int(data["max_depth"])
        self.trained = data.get("trained", {})
        self.version = f"{self.id}+{hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]}"

    @classmethod
    def load(cls, name: str = "student-1") -> "Student":
        return cls(json.loads((Path(__file__).parent / f"{name}.json").read_text(encoding="utf-8")))

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """모든 트리를 한꺼번에 내려가며 (트리 × 후보) 잎 값을 더한다."""
        n = len(X)
        if n == 0:
            return np.zeros(0)
        T = len(self.feature)
        node = np.zeros((T, n), dtype=np.int64)
        cols = np.arange(n)[None, :]
        tree = np.arange(T)[:, None]
        for _ in range(self.depth):
            left = self.left[tree, node]
            go_left = X[cols, self.feature[tree, node]] <= self.cut[tree, node]
            node = np.where(left < 0, node, np.where(go_left, left, self.right[tree, node]))
        raw = self.baseline + self.value[tree, node].sum(0)
        return 1.0 / (1.0 + np.exp(-raw))

    def select(self, extractor, note_id: str, note_version: int, text: str) -> dict[str, list[Mention]]:
        """개념 key → 언급들 (확률이 높은 순, 같으면 key 순). 응답에 실을 모양 그대로 (UTF-16 위치, 점수 = 2 × 확률)."""
        text, rows_ = rows(extractor, text)
        if not rows_:
            return {}
        p = self.probabilities(np.array([r["x"] for r in rows_], dtype=np.float64))
        p = np.round(p, 6)                                              # 플랫폼별 부동소수 끝자리 차이가 경계에서 갈리지 않게
        idx = sorted(suppress_nested(rows_, p, self.threshold), key=lambda i: (-p[i], rows_[i]["key"]))[:MAX_CONCEPTS]
        u16 = Utf16Index(text)
        out = {}
        for i in idx:
            r = rows_[i]
            spans = sorted(set(r["spans"]))[:MAX_MENTIONS]
            out[r["key"]] = [Mention(span=Span(note_id, note_version, u16(s), u16(e)), surface=text[s:e], key=r["key"], method=Method.TFIDF,
                                     method_version=self.version, score=round(2.0 * float(p[i]), 4)) for s, e in spans]
        return out
