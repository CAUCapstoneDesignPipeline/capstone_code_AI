"""결정적 개념 추출 (design §4①).

Kiwi 명사구 + 마크다운 구조 → 후보, 고정 IDF 의 TF-IDF + 구조 가중치 → 점수, 임계값 → 선택.
같은 원문·설정·IDF 면 결과가 항상 같다. 무작위성이 없고, 모든 반복은 정렬된 순서로 한다.
"""

import hashlib
import json
import math
import re
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path

from .config import ExtractConfig
from .markdown import BOLD, HEADING, TABLE, Utf16Index, mask, nfc, region_parts, regions, strip_inline_marks
from .schema import Mention, Method, Span

NOUN_TAGS = {"NNG", "NNP", "SL", "SH"}          # 명사구의 몸통이 될 수 있는 품사
INNER_TAGS = NOUN_TAGS | {"SN", "XSN", "XPN", "W_SERIAL"}  # 명사구 안에만 올 수 있는 품사
CONNECTORS = {"-", "_", "/", "+"}                # 명사 사이를 잇는 기호 (Bellman-Ford, TCP/IP)
SEPARATOR_WORDS = {"vs", "versus"}               # 영어 명사로 읽히지만 두 개념을 가르는 말
PREDICATE_SUFFIX = {"XSV", "XSA"}                # 명사를 동사·형용사로 만드는 접미사 (-하다, -되다)
STRUCT_COVER = 0.6                               # 후보가 구조 조각의 이만큼 이상을 덮어야 구조 가중치를 받는다
# 구조 구간 조각이 통째로 개념이 되려면 이 품사로만 이뤄져야 한다 (조사·어미·동사가 있으면 문장이다)
REGION_TAGS = INNER_TAGS | {"SO", "SS", "SW", "SP", "W_EMOJI", "NNB"}

_EDGE_PUNCT = "\"'“”‘’()[]{}<>:;,.!?·"


def concept_key(surface: str) -> str:
    """정규화 키: NFC, 소문자, 공백 하나로, 양 끝 기호 제거. 'WAL' 과 'wal' 은 같은 key."""
    s = re.sub(r"\s+", " ", nfc(surface)).strip().strip(_EDGE_PUNCT).strip()
    return s.lower()


DETACHED_PARTICLES = {"은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "로", "으로", "에서", "만"}


def _after_symbol(text: str, i: int) -> bool:
    """i 앞 (같은 줄의 공백은 건너뜀) 이 글자·숫자가 아닌 기호다: ), `, $, * 등."""
    j = i - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    return j >= 0 and text[j] != "\n" and not text[j].isalnum()


_Tok = namedtuple("_Tok", "form tag start len")


class StaleIdfError(RuntimeError):
    """IDF 파일이 지금 추출 설정과 다른 설정으로 만들어졌다."""


@dataclass(frozen=True)
class Candidate:
    start: int       # Python 문자 위치 (출력할 때 UTF-16 으로 바꾼다)
    end: int
    surface: str
    key: str
    struct: float    # 이 후보를 덮는 구조 구간의 가장 큰 가중치 (없으면 0)


class Idf:
    """고정 기준 말뭉치의 IDF. 노트가 늘 때마다 다시 계산하지 않고 버전을 붙여 관리한다."""

    def __init__(self, df: dict[str, int], n_docs: int, version: str, config_version: str | None = None):
        self.df, self.n_docs, self.version, self.config_version = df, n_docs, version, config_version

    def __call__(self, key: str) -> float:
        return math.log((self.n_docs + 1) / (self.df.get(key, 0) + 1)) + 1.0

    def save(self, path: Path, config_version: str):
        data = {"version": self.version, "config_version": config_version, "n_docs": self.n_docs,
                "df": dict(sorted(self.df.items()))}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Idf":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["df"], data["n_docs"], data["version"], data.get("config_version"))

    @classmethod
    def empty(cls) -> "Idf":
        return cls({}, 0, "none")


class ConceptExtractor:
    def __init__(self, config: ExtractConfig | None = None, idf: Idf | None = None,
                 cache_dir: Path | None = None):
        from kiwipiepy import Kiwi  # 무거운 모델 로딩은 추출기를 만들 때 한 번만

        self.config = config or ExtractConfig()
        self.idf = idf or Idf.empty()
        self.cache_dir = cache_dir
        self.kiwi = Kiwi()
        self.config_version = self.config.version()
        # IDF 의 후보 목록은 추출 설정에 따라 다르다. 다른 설정으로 만든 IDF 를 쓰면 점수가 조용히 어긋난다.
        if self.idf.config_version and self.idf.config_version != self.config_version:
            raise StaleIdfError(f"IDF {self.idf.version} 는 설정 {self.idf.config_version} 로 만들어졌다 "
                                f"(지금 {self.config_version}). `ai concepts build-idf` 로 다시 만든다.")

    @property
    def method_version(self) -> str:
        return f"cfg-{self.config_version}+idf-{self.idf.version}"

    # ---- 후보 ----

    def candidates(self, text: str) -> list[Candidate]:
        text = nfc(text)
        masked = mask(text)
        regs = regions(masked)
        weight = {HEADING: self.config.heading_weight, BOLD: self.config.bold_weight,
                  TABLE: self.config.table_weight}
        spans: dict[tuple[int, int], str] = {}

        # 1) 명사구: 붙어 있거나 공백 하나·연결 기호로 이어진 명사를 묶는다
        run: list = []

        def flush():
            body = [t for t in run if t.tag in INNER_TAGS]
            while body and body[0].tag == "XSN":                  # 접미사로 시작할 수는 없다
                body.pop(0)
            nouns = [t for t in body if t.tag in NOUN_TAGS]
            for t in nouns:                                       # 낱말 하나
                spans[(t.start, t.start + t.len)] = "token"
            # 복합 명사구 (페이지 캐시, 0-1 Knapsack, 안정+성)
            if nouns and len(body) > 1 and len(nouns) <= self.config.max_phrase_tokens:
                spans[(body[0].start, body[-1].start + body[-1].len)] = "phrase"
            run.clear()

        prev_end = None
        # 가린 구간(코드·수식·주소)을 공백 대신 같은 길이의 'X' 로 분석기에 넘긴다. 공백이면 "$V$ 가 다항" 의
        # '가 다' 를 동사 '가다' 로 읽는다. 가린 구간에 걸친 토큰은 개념이 아니라 명사구 경계로 쓴다.
        hidden = [m == " " and c not in " \t\n" for m, c in zip(masked, text)]
        clean, pos = strip_inline_marks("".join("X" if h else m for h, m in zip(hidden, masked)))
        for t in self._tokens(clean, pos):
            if any(hidden[t.start:t.start + t.len]):
                flush()
                prev_end = None
                continue
            # `x`가, $n$을 처럼 가린 코드·수식 바로 뒤의 조사는 분석기가 명사로 읽는다. 명사구를 끊는다
            if (t.tag.startswith(("NN", "XPN")) and t.form in DETACHED_PARTICLES
                    and _after_symbol(text, t.start)):
                flush()
                prev_end = None
                continue
            if t.form.lower().rstrip(".") in SEPARATOR_WORDS:   # "Static Scope vs. Dynamic Scope" 의 vs 는 명사가 아니다
                flush()
                prev_end = None
                continue
            is_conn = t.tag == "SO" and t.form in CONNECTORS
            if t.tag in INNER_TAGS or (is_conn and run):
                gap = masked[prev_end:t.start] if prev_end is not None else ""
                if run and gap not in ("", " "):
                    flush()
                if not (is_conn and not run):
                    run.append(t)
                prev_end = t.start + t.len
            else:
                # 명사 바로 뒤에 하다·되다류 접미사가 붙으면 서술어로 쓰인 것이다 (포함한, 저장된, 수정하면)
                if t.tag in PREDICATE_SUFFIX and run and prev_end == t.start:
                    run.pop()
                flush()
                prev_end = None
        flush()

        # 2) 구조 구간 조각: 명사로만 된 짧은 제목·굵은 글씨·표 첫 칸은 통째로 후보 (Banker's Algorithm)
        parts = []  # (start, end, 가중치)
        for s, e, kind in regs:
            for ps, pe in region_parts(text, s, e):
                parts.append((ps, pe, weight[kind]))
                piece = text[ps:pe]
                if len(piece) > self.config.max_surface_chars or re.search(r"[*`_\[\]]", piece):
                    continue  # 너무 길거나 마크다운 기호가 남은 조각은 명사구 쪽에 맡긴다
                toks = self.kiwi.tokenize(piece)
                if toks and toks[0].form in DETACHED_PARTICLES and _after_symbol(text, ps):
                    continue  # "(PFR)가 증가" 를 나눈 조각 "가 증가" 의 '가' 는 조사다
                if toks and all(t.tag in REGION_TAGS for t in toks) and any(t.tag in NOUN_TAGS for t in toks):
                    spans[(ps, pe)] = "region"

        # 3) 가장 긴 것 우선: 더 긴 후보 안에만 있는 낱말은 따로 세지 않는다 (buffer pool 안의 pool)
        ordered = sorted(spans)
        kept = []
        for s, e in ordered:
            if spans[(s, e)] == "token" and any(
                    os_ <= s and e <= oe and (os_, oe) != (s, e) for os_, oe in ordered if spans[(os_, oe)] != "token"):
                continue
            kept.append((s, e))

        out = []
        for s, e in kept:
            surface = text[s:e]
            key = concept_key(surface)
            if not self._keep(key):
                continue
            # 구조 가중치는 후보가 그 조각 자체이거나 대부분을 덮을 때만 준다.
            # 문장 전체를 굵게 쓴 경우 그 안의 모든 명사가 가중치를 받지 않게.
            struct = max((w for ps, pe, w in parts
                          if ps <= s and e <= pe and (e - s) >= STRUCT_COVER * (pe - ps)), default=0.0)
            out.append(Candidate(s, e, surface, key, struct))
        return out

    def _tokens(self, clean: str, pos: list[int]):
        """기호를 뺀 글의 토큰을 원문 위치로 돌려 낸다 (start, len 만 원문 기준으로 바꾼다)."""
        for t in self.kiwi.tokenize(clean):
            start = pos[t.start]
            end = pos[t.start + t.len - 1] + 1 if t.len else start
            yield _Tok(t.form, t.tag, start, end - start)

    def _keep(self, key: str) -> bool:
        c = self.config
        if not (c.min_surface_chars <= len(key) <= c.max_surface_chars):
            return False
        if key in c.stopwords or re.sub(r"\s*\d+$", "", key) in c.stopwords:   # 참고1, 예시 2
            return False
        words = [w for w in re.split(r"[\s\-_/]+", key) if w and not w.isdigit()]
        if words and all(w in c.stopwords for w in words):                  # 핵심 아이디어, 문제 정의, step 1
            return False
        if not re.search(r"[A-Za-z가-힣一-鿿]", key):          # 글자 없는 숫자·기호
            return False
        if re.fullmatch(r"[a-z]", key) or re.fullmatch(r"[\d\s.\-]+", key):
            return False
        if re.fullmatch(r"[a-z]\d*|\d+[a-z]?", key):            # 변수 이름 같은 것 (n, x1, 2d)
            return False
        return True

    # ---- 점수와 선택 ----

    def extract(self, note_id: str, text: str, note_version: int = 1) -> list[Mention]:
        text = nfc(text)
        cached = self._cache_get(text)
        if cached is not None:
            return [Mention.from_dict({**m, "span": {**m["span"], "note_id": note_id,
                                                      "note_version": note_version}}) for m in cached]

        cands = self.candidates(text)
        by_key: dict[str, list[Candidate]] = defaultdict(list)
        for c in cands:
            by_key[c.key].append(c)

        tfidf = {k: (1 + math.log(len(v))) * self.idf(k) for k, v in by_key.items()}
        top = max(tfidf.values(), default=0.0) or 1.0
        scores = {}
        for k in sorted(by_key):
            struct = max(c.struct for c in by_key[k])
            scores[k] = round(tfidf[k] / top + struct, 4)   # 반올림해 부동소수 오차로 결과가 바뀌지 않게
        chosen = {k for k, s in scores.items() if s >= self.config.threshold}

        u16 = Utf16Index(text)
        mentions = []
        for c in cands:
            if c.key not in chosen:
                continue
            mentions.append(Mention(
                span=Span(note_id, note_version, u16(c.start), u16(c.end)),
                surface=c.surface, key=c.key,
                method=Method.MARKDOWN if c.struct > 0 else Method.TFIDF,
                method_version=self.method_version, score=scores[c.key],
            ))
        mentions.sort(key=lambda m: (m.span.start, m.span.end, m.surface))
        self._cache_put(text, mentions)
        return mentions

    # ---- 캐시: 같은 원문·설정·IDF 면 다시 계산하지 않는다 ----

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(f"{self.method_version}\n{text}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _cache_get(self, text: str):
        p = self._cache_path(text)
        if p and p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, text: str, mentions: list[Mention]):
        p = self._cache_path(text)
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps([m.to_dict() for m in mentions], ensure_ascii=False), encoding="utf-8")


def build_idf(extractor: ConceptExtractor, texts: dict[str, str]) -> Idf:
    """기준 말뭉치에서 문서 빈도를 센다. 버전 = 말뭉치 원문 해시들 + 설정 버전의 해시."""
    df: dict[str, int] = defaultdict(int)
    hashes = []
    for note_id in sorted(texts):
        text = nfc(texts[note_id])
        hashes.append(hashlib.sha256(text.encode()).hexdigest())
        for key in sorted({c.key for c in extractor.candidates(text)}):
            df[key] += 1
    version = hashlib.sha256("\n".join(sorted(hashes) + [extractor.config_version]).encode()).hexdigest()[:12]
    return Idf(dict(df), len(texts), version, extractor.config_version)


def summarize(mentions: list[Mention]) -> list[dict]:
    """key 별 요약: 대표 표기(가장 많이 쓰인 표기, 같으면 사전순), 언급 수, 점수, 방법."""
    groups: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        groups[m.key].append(m)
    rows = []
    for key, ms in groups.items():
        counts: dict[str, int] = defaultdict(int)
        for m in ms:
            counts[m.surface] += 1
        label = sorted(counts, key=lambda s: (-counts[s], s))[0]
        rows.append({"key": key, "label": label, "mentions": len(ms), "score": ms[0].score,
                     "markdown": any(m.method is Method.MARKDOWN for m in ms)})
    return sorted(rows, key=lambda r: (-r["score"], r["key"]))
