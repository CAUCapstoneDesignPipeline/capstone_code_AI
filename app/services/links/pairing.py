"""새 지식을 만들 키워드 요소(개념) 쌍 후보를 찾는다 (결정적, LLM 호출 없음).

새 지식 = 키워드 A 의 요소 P + 키워드 B 의 요소 Q + 외부 지식. 여기서는 P(이 노트의 요소)와 Q(다른 노트·다른 키워드의 요소)를 잇는다.
실제 노트 275쌍을 두 번 사람 대신 AI 두 명이 눈가림으로 라벨링해서 얻은 결론을 따른다:
  - 신호는 하나로는 약하다. 공유 낱말(2개 이상) · 공통 키워드 언급 · 임베딩 근접 셋을 합치고, 겹칠수록 먼저 둔다.
  - 코사인 0.5 미만은 쓸모 있는 쌍이 하나도 없었다 (0 / 89). 임베딩이 있으면 0.5 를 바닥으로 둔다.
  - 요소가 혼자 읽을 수 있어야 한다 (너무 짧거나 이어 붙는 말로 시작하거나 표 칸인 것은 뺀다).
  - 후보는 적어도 3~16% 만 쓸모 있었다. 이 후보를 LLM 선별(pair_screen) 이 거른다.

비용 (요소 E 개, 이 노트의 요소 A 개). 모든 쌍(E²/2)을 비교하지 않는다 — 요소마다 신호가 가리키는 몇십 개만 본다:
  - 색인은 O(글자 수) 한 번: 공유 낱말 역색인, 키워드 이름 trie 로 "어느 요소가 어느 키워드를 말하는지" 를 글 하나당 한 번 훑어 찾는다
    (요소마다 모든 키워드 이름을 훑으면 A·K 번 부분 문자열 검사가 든다. 실제 노트에서 전체 3,318 요소 기준 6.9초 → 0.75초)
  - 공유 낱말 O(낱말 수 · 25), 공통 키워드 O(말한 키워드 수 · 40): 낱말·키워드 빈도 상한이 있어 말뭉치가 커져도 늘지 않는다
  - 임베딩 근접은 A × E 행렬곱 한 번 + argpartition (정렬하지 않는다). E × E 행렬은 만들지 않는다. 근사 색인(군집 IVF)도 해 봤지만
    실제 임베딩에서는 이웃 10개의 90% 를 찾으려면 전체의 26~46% 를 훑어야 해서 (재현율 0.63~0.95) 정확한 검색이 낫다.
    E 가 수십만이 되면 이 자리(_nearest_all)에 HNSW 같은 그래프 색인을 꽂는다
  - 새 지식 만들기(LLM + 웹 검색) 는 후보 수가 아니라 고정한 개수(상위 k) 나 사용자가 이은 한 쌍만 한다 — 이어진 쌍마다 하지 않는다
"""

import hashlib
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MD = re.compile(r"\*\*|__|`")
START = re.compile(r"^(그래서|그러면|그런데|하지만|그리고|따라서|즉|또|또한|이|그|저|이런|그런|이것|그것|이를|그걸)\s")
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[가-힣]{3,}")
MIN_CHARS, MAX_CHARS = 20, 300
TOKEN_DF = (2, 25)             # 너무 흔하거나 한 번뿐인 낱말은 연결고리가 못 된다
PIVOT_DF = (2, 40)
COSINE_FLOOR = 0.5
MODEL, REVISION = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"


def clean(text: str) -> str:
    return MD.sub("", text).strip()


def gate(text: str) -> bool:
    """혼자 읽을 수 있는 설명인가."""
    return MIN_CHARS <= len(text) <= MAX_CHARS and text.count("|") < 2 and "://" not in text and not START.match(text)


@dataclass(frozen=True)
class Element:
    idx: int
    keyword_id: str
    keyword: str
    aliases: tuple[str, ...]
    note: str
    note_id: str
    concept_id: str
    text: str                       # 마크다운 기호를 뺀 설명
    confidence: float | None
    start: int                      # 노트 안 위치 (UTF-16)
    end: int
    quote: str


@dataclass(frozen=True)
class Candidate:
    a: Element                      # 이 노트의 요소
    b: Element                      # 다른 노트의 요소
    channels: tuple[str, ...]       # terms | pivot | embed
    cos: float | None
    shared: tuple[str, ...] = ()    # terms: 공유 낱말
    via: tuple[str, ...] = ()       # pivot: 둘 다 언급하는 키워드

    @property
    def score(self) -> float:
        return len(self.channels) + (self.cos or 0.0)


def elements_from(keywords, meta) -> list[Element]:
    out = []
    for k in keywords:
        for c in k.concepts:
            src = c.sources[0]
            m = meta[c.conceptId]
            out.append(Element(len(out), k.keywordId, k.name, tuple(k.aliases), m["note"], src.noteId, c.conceptId,
                               clean(c.explanation), m.get("confidence"), src.span.startOffset, src.span.endOffset, src.span.quote))
    return out


class PairingIndex:
    def __init__(self, elements: list[Element], vectors=None):
        self.el, self.vec = elements, vectors
        self.ok = [gate(e.text) for e in elements]
        self.low = [e.text.lower() for e in elements]
        self.tokens = [set(TOKEN.findall(t)) if ok else set() for t, ok in zip(self.low, self.ok)]
        df = Counter(w for ts in self.tokens for w in ts)
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, ts in enumerate(self.tokens):
            for w in ts:
                if TOKEN_DF[0] <= df[w] <= TOKEN_DF[1]:
                    self.inv[w].append(i)
        self.kw_names: dict[str, list[str]] = {}
        self.kw_label = {e.keyword_id: e.keyword for e in elements}
        for e in elements:
            names = {s.lower() for s in (e.keyword, *e.aliases) if len(s) >= 3}
            self.kw_names.setdefault(e.keyword_id, sorted(names))
        self.name_tokens = [set().union(*(set(TOKEN.findall(n)) for n in self.kw_names[e.keyword_id])) if self.kw_names[e.keyword_id] else set()
                            for e in elements]
        self._index_mentions()
        if vectors is not None:
            import numpy as np
            codes = lambda xs: np.unique(xs, return_inverse=True)[1]
            self._ok_arr = np.array(self.ok)
            self._kid_code, self._note_code = codes([e.keyword_id for e in elements]), codes([e.note for e in elements])

    def _index_mentions(self):
        """키워드 이름·별칭이 설명에 나오는지를 글 하나당 한 번의 훑기로 모두 찾는다 (이름 trie). 요소마다 모든 이름을 훑지 않는다."""
        trie: dict = {}
        for kid, names in self.kw_names.items():
            for n in names:
                node = trie
                for ch in n:
                    node = node.setdefault(ch, {})
                node.setdefault(None, set()).add(kid)
        self.elem_kws: list[set[str]] = [set() for _ in self.el]
        self.mentioners: dict[str, list[int]] = defaultdict(list)
        for i, text in enumerate(self.low):
            if not self.ok[i]:
                continue
            found: set[str] = set()
            for s in range(len(text)):
                node = trie.get(text[s])
                t = s + 1
                while node is not None:
                    if None in node:
                        found |= node[None]
                    if t >= len(text):
                        break
                    node = node.get(text[t])
                    t += 1
            found.discard(self.el[i].keyword_id)                   # 자기 키워드는 다른 키워드를 잇는 고리가 아니다
            self.elem_kws[i] = found
            for kid in found:
                self.mentioners[kid].append(i)

    def _valid(self, i: int, j: int) -> bool:
        a, b = self.el[i], self.el[j]
        return self.ok[i] and self.ok[j] and a.keyword_id != b.keyword_id and a.note != b.note

    def anchors(self, note: str, limit: int = 40) -> list[int]:
        """이 노트의 요소 중 혼자 읽을 수 있는 것. 키워드마다 하나, 신뢰도 순."""
        best: dict[str, int] = {}
        for e in self.el:
            if e.note == note and self.ok[e.idx]:
                cur = best.get(e.keyword_id)
                if cur is None or (e.confidence or 0) > (self.el[cur].confidence or 0):
                    best[e.keyword_id] = e.idx
        return sorted(best.values(), key=lambda i: (-(self.el[i].confidence or 0), i))[:limit]

    def _nearest_all(self, anchors: list[int], k: int) -> dict[int, list[int]]:
        """anchors 마다 가까운 다른 노트·다른 키워드 요소 k 개. 행렬곱 한 번(256개씩)으로 구하고 정렬 없이 k 개만 뽑는다."""
        import numpy as np
        out: dict[int, list[int]] = {}
        anchors = [i for i in anchors if self.ok[i]]
        k = min(k, len(self.el) - 1)
        if self.vec is None or not anchors or k < 1:
            return out
        for s in range(0, len(anchors), 256):
            chunk = anchors[s:s + 256]
            sims = self.vec[chunk] @ self.vec.T                                  # (chunk, E)
            bad = (~self._ok_arr[None, :] | (self._kid_code[None, :] == self._kid_code[chunk][:, None])
                   | (self._note_code[None, :] == self._note_code[chunk][:, None]))
            sims[bad] = -np.inf
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for r, i in enumerate(chunk):
                out[i] = [int(j) for j in top[r] if np.isfinite(sims[r, j])]
        return out

    def anchors(self, note: str, limit: int = 40) -> list[int]:
        """이 노트의 요소 중 혼자 읽을 수 있는 것. 키워드마다 하나, 신뢰도 순."""
        best: dict[str, int] = {}
        for e in self.el:
            if e.note == note and self.ok[e.idx]:
                cur = best.get(e.keyword_id)
                if cur is None or (e.confidence or 0) > (self.el[cur].confidence or 0):
                    best[e.keyword_id] = e.idx
        return sorted(best.values(), key=lambda i: (-(self.el[i].confidence or 0), i))[:limit]

    def _nearest(self, i: int, k: int, exact: bool) -> list[int]:
        """요소 i 와 가까운 다른 노트·다른 키워드의 요소 k 개. 군집 색인으로 √E 개 군집 중 가까운 몇 개만 훑는다."""
        import numpy as np
        if self._ivf is None and not exact:
            self._ivf = _IVF(self.vec)
        q = self.vec[i]
        kid_i, note_i = self.el[i].keyword_id, self.el[i].note
        cand = np.arange(len(self.el)) if exact else None
        probe = 4
        while True:
            if not exact:
                cand = self._ivf.probe(q, probe)
            sims = self.vec[cand] @ q
            order = np.argsort(-sims)
            out = []
            for pos in order:
                j = int(cand[pos])
                if j != i and self.ok[j] and self.el[j].keyword_id != kid_i and self.el[j].note != note_i:
                    out.append(j)
                    if len(out) >= k:
                        return out
            if exact or probe >= self._ivf.nlist:
                return out
            probe = min(self._ivf.nlist, probe * 2)             # 쓸 수 있는 이웃이 모자라면 더 넓게 훑는다

    def candidates(self, anchors: list[int], per_anchor: int = 3, per_partner: int = 1, limit: int = 12,
                   neighbors: int = 10) -> tuple[list[Candidate], dict]:
        near = self._nearest_all(anchors, neighbors)
        found: dict[tuple[int, int], dict] = {}

        def add(i: int, j: int, ch: str, **extra):
            if self._valid(i, j):
                f = found.setdefault((i, j), {"channels": [], "shared": (), "via": ()})
                if ch not in f["channels"]:
                    f["channels"].append(ch)
                for k, v in extra.items():
                    f[k] = tuple(dict.fromkeys((*f[k], *v)))

        counts = Counter()
        for i in anchors:
            shared: dict[int, list[str]] = defaultdict(list)           # terms
            for w in self.tokens[i]:
                if w in self.inv and w not in self.name_tokens[i]:
                    for j in self.inv[w]:
                        if j != i and w not in self.name_tokens[j]:
                            shared[j].append(w)
            for j, ws in shared.items():
                if len(ws) >= 2:
                    add(i, j, "terms", shared=tuple(sorted(ws)))
            for kid in self.elem_kws[i]:                                # pivot: 이 요소가 말하는 다른 키워드를 같이 말하는 요소
                ms = self.mentioners[kid]
                if PIVOT_DF[0] <= len(ms) <= PIVOT_DF[1]:
                    for j in ms:
                        add(i, j, "pivot", via=(self.kw_label[kid],))
            for j in near.get(i, []):                                   # embed
                add(i, j, "embed")
        for f in found.values():
            for ch in f["channels"]:
                counts[ch] += 1
        out = []
        for (i, j), f in found.items():
            cos = float(self.vec[i] @ self.vec[j]) if self.vec is not None else None
            if cos is not None and cos < COSINE_FLOOR:
                continue
            out.append(Candidate(self.el[i], self.el[j], tuple(f["channels"]), None if cos is None else round(cos, 3),
                                 f["shared"], f["via"]))
        counts["union"] = len(found)
        counts["afterFloor"] = len(out)
        out.sort(key=lambda c: (-c.score, c.a.idx, c.b.idx))
        picked, per_a, per_b, seen = [], Counter(), Counter(), set()
        for c in out:
            ta, tb = c.a.text, c.b.text                       # 같은 문장을 다른 키워드가 가리키는 중복은 한 번만
            if per_a[ta] >= per_anchor or per_b[tb] >= per_partner or (ta, tb) in seen or (c.a.keyword_id, c.b.keyword_id) in seen:
                continue
            picked.append(c)
            per_a[ta] += 1
            per_b[tb] += 1
            seen.add((ta, tb))
            seen.add((c.a.keyword_id, c.b.keyword_id))
            if len(picked) >= limit:
                break
        return picked, dict(counts)


class EmbeddingStore:
    """요소 설명 임베딩 (bge-m3, 로컬). 한 번 계산한 것은 디스크에 두고 다시 쓴다. 모델 파일이 이미 받아져 있을 때만 쓴다 (내려받지 않는다)."""

    def __init__(self, path: Path):
        self.path, self._lock, self._model, self._tried = path, threading.Lock(), None, False

    def _load_model(self):
        if self._tried:
            return self._model
        self._tried = True
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
            base = ["*.json", "*.txt", "*.model", "1_Pooling/*"]           # app/services/embeddings 가 받는 파일과 같다
            try:
                snap = snapshot_download(MODEL, revision=REVISION, allow_patterns=[*base, "model.safetensors"], local_files_only=True)
                if not (Path(snap) / "model.safetensors").exists():
                    raise FileNotFoundError("model.safetensors")
            except Exception:
                snap = snapshot_download(MODEL, revision=REVISION, allow_patterns=[*base, "pytorch_model.bin"], local_files_only=True)
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
            self._model = SentenceTransformer(snap, device=device)
            self._model.max_seq_length = 192
        except Exception:
            self._model = None
        return self._model

    def vectors(self, texts: list[str]):
        """texts 의 정규화 벡터 (n, d). 모델이 없으면 None."""
        import numpy as np
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        with self._lock:
            have: dict[str, "np.ndarray"] = {}
            if self.path.exists():
                z = np.load(self.path, allow_pickle=False)
                have = dict(zip(z["keys"].tolist(), z["mat"]))
            missing = sorted({k: t for k, t in zip(keys, texts) if k not in have}.items())
            if missing:
                model = self._load_model()
                if model is None:
                    return None
                new = model.encode([t for _, t in missing], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
                have.update({k: v.astype(np.float32) for (k, _), v in zip(missing, new)})
                self.path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.path, keys=np.array(list(have)), mat=np.stack(list(have.values())))
            return np.stack([have[k] for k in keys])
