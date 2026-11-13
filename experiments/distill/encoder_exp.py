"""문맥을 읽는 작은 인코더(multilingual-e5-small)가 학생(부스팅 트리 + 특징 62개)을 더 낫게 하는지 잰다.

  uv run --env-file .env python experiments/distill/encoder_exp.py [--max-notes 0]

노트 전체를 인코더에 넣어(512토큰 창, 절반씩 겹침) 토큰마다 문맥 벡터를 얻고, 후보의 벡터 = 그 후보 언급(앞 5개)의 토큰 평균을 다시 평균한 것이다.
앞서 해 본 표기 임베딩(표기 문자열만 따로 넣은 것)과 달리 문장 속에서 읽은 벡터다. 벡터로는
  probe : 벡터만 보고 개념인지 고르는 로지스틱 탐침의 확률 (노트 단위 교차로 학습해 새는 것을 막는다)
  sim   : 후보 벡터와 노트 평균 벡터의 코사인 (주제와 가까운가)
을 만들어 트리에 특징으로 더한다. 학습 자료·정답·채점·임계값 고르기는 nn_exp.py 와 같다 (3 시드 평균).
"""

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SP = Path("/private/tmp/claude-501/-Users-klee9-capstone/6ecbcd7c-aceb-429f-97e6-1e37e2c2da24/scratchpad")
CACHE = SP / "encoder_cache.pkl"
MODEL = "intfloat/multilingual-e5-small"
MAX_MENTIONS = 5


class Encoder:
    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def tokens(self, text):
        """(토큰별 문맥 벡터 (L, D), 토큰 글자 구간 (L, 2))."""
        torch = self.torch
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], np.array(enc["offset_mapping"], int).reshape(-1, 2)
        L = len(ids)
        if L == 0:
            return np.zeros((0, self.dim), np.float32), offs
        wins, a = [], 0
        while True:
            b = min(a + 510, L)
            wins.append((a, b))
            if b == L:
                break
            a += 255
        acc, cnt = np.zeros((L, self.dim), np.float32), np.zeros(L, np.float32)
        for i in range(0, len(wins), 8):
            chunk = wins[i:i + 8]
            width = max(b - a for a, b in chunk) + 2
            x = torch.full((len(chunk), width), self.tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, (a, b) in enumerate(chunk):
                seq = [self.tok.cls_token_id, *ids[a:b], self.tok.sep_token_id]
                x[j, :len(seq)] = torch.tensor(seq)
                m[j, :len(seq)] = 1
            with torch.inference_mode():
                h = self.model(input_ids=x.to(self.device), attention_mask=m.to(self.device)).last_hidden_state.float().cpu().numpy()
            for j, (a, b) in enumerate(chunk):
                acc[a:b] += h[j, 1:1 + b - a]
                cnt[a:b] += 1
        return acc / cnt[:, None], offs


def candidate_vectors(H, offs, rs):
    """후보마다 (벡터 (D,) 정규화, 노트 평균과의 코사인)."""
    D = H.shape[1]
    if not len(H) or not rs:
        return np.zeros((len(rs), D), np.float32), np.zeros(len(rs), np.float32)
    C = np.vstack([np.zeros((1, D), np.float64), np.cumsum(H, 0, dtype=np.float64)])
    note = H.mean(0)
    note /= np.linalg.norm(note) + 1e-9
    V = np.zeros((len(rs), D), np.float32)
    for k, r in enumerate(rs):
        acc = np.zeros(D)
        spans = r["spans"][:MAX_MENTIONS]
        for s, e in spans:
            i0 = int(np.searchsorted(offs[:, 1], s, "right"))
            i1 = max(int(np.searchsorted(offs[:, 0], e, "left")), i0 + 1)
            i0, i1 = min(i0, len(H) - 1), min(i1, len(H))
            acc += (C[i1] - C[i0]) / max(1, i1 - i0)
        v = acc / max(1, len(spans))
        V[k] = v / (np.linalg.norm(v) + 1e-9)
    return V, V @ note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-notes", type=int, default=0, help="학습 노트 수 상한 (시험용)")
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from app.services.concepts.student import rows, suppress_nested
    from experiments.distill.features import teacher_concepts
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    enc = None
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    stats = {"enc_s": [], "chars": []}

    def vectors(t2, rs):
        nonlocal enc
        key = hashlib.sha1(t2.encode()).hexdigest()
        if key not in cache:
            enc = enc or Encoder()
            t0 = time.time()
            H, offs = enc.tokens(t2)
            V, sim = candidate_vectors(H, offs, rs)
            stats["enc_s"].append(time.time() - t0)
            stats["chars"].append(len(t2))
            cache[key] = (V.astype(np.float16), sim)
        V, sim = cache[key]
        return V.astype(np.float32), sim

    web = {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes-web").glob("*.md"))}
    train, t0 = [], time.time()
    for i, (_, text) in enumerate([*teacher_notes().items(), *web.items()]):
        if args.max_notes and len(train) >= args.max_notes:
            break
        g = teacher_concepts(text)
        if g is None:
            continue
        forms = frozenset().union(*[k for _, k in g]) if g else frozenset()
        t2, rs = rows(ex, text)
        if not rs:
            continue
        V, sim = vectors(t2, rs)
        y = np.array([int(bool({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"][:3])} & forms)) for r in rs])
        train.append((np.array([r["x"] for r in rs], float), V, sim, y))
        if len(train) % 25 == 0:
            print(f"  train notes {len(train)} ({time.time() - t0:.0f}s)", flush=True)
            pickle.dump(cache, open(CACHE, "wb"))
    X = np.vstack([t[0] for t in train])
    V = np.vstack([t[1] for t in train])
    S = np.concatenate([t[2] for t in train])
    y = np.concatenate([t[3] for t in train])
    groups = np.concatenate([[i] * len(t[3]) for i, t in enumerate(train)])
    print(f"training: {len(train)} notes, {len(y)} candidates, {y.mean():.1%} positive", flush=True)

    data = sets_available(["dev", "test", "unseen"])
    prepared = {}
    for name, (gold, texts) in data.items():
        per = {}
        for n in sorted(gold):
            t2, rs = rows(ex, texts[n])
            if rs:
                Ve, Se = vectors(t2, rs)
                per[n] = (t2, rs, np.array([r["x"] for r in rs], float), Ve, Se)
            else:
                per[n] = (t2, rs, None, None, None)
        prepared[name] = (gold, per)
    pickle.dump(cache, open(CACHE, "wb"))
    if stats["enc_s"]:
        print(f"encoder: {len(stats['enc_s'])} notes encoded, mean {np.mean(stats['enc_s']):.2f}s/note ({np.mean(stats['chars']):.0f} chars)", flush=True)

    # 탐침: 노트 단위 교차로 학습한 확률(학습용 특징) + 전체로 학습한 탐침(평가용)
    oof = np.zeros(len(y))
    t0 = time.time()
    for tr, te in GroupKFold(5).split(V, y, groups):
        oof[te] = LogisticRegression(C=10, max_iter=150).fit(V[tr], y[tr]).predict_proba(V[te])[:, 1]
    probe = LogisticRegression(C=10, max_iter=150).fit(V, y)
    print(f"probe trained ({time.time() - t0:.0f}s): OOF mean prob pos {oof[y == 1].mean():.2f} vs neg {oof[y == 0].mean():.2f}", flush=True)

    def evaluate(name, fn, thr):
        gold, per = prepared[name]
        sc = {}
        for n, (t2, rs, x, Ve, Se) in per.items():
            ext = []
            if rs:
                p = fn(x, Ve, Se)
                for i in suppress_nested(rs, p, thr):
                    r = rs[i]
                    ext.append(Extracted(r["surface"], frozenset({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"])}), float(p[i])))
            sc[n] = score(gold[n], ext)
        R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
        P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
        return R, P, 2 * R * P / (R + P + 1e-9)

    def gbt(cols, seed):
        m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=30, l2_regularization=1.0, random_state=seed).fit(cols, y)
        return m

    variants = {
        "student (62 features)": (lambda: X, lambda x, v, s: x),
        "+ probe": (lambda: np.column_stack([X, oof]), lambda x, v, s: np.column_stack([x, probe.predict_proba(v)[:, 1]])),
        "+ probe + sim": (lambda: np.column_stack([X, oof, S]), lambda x, v, s: np.column_stack([x, probe.predict_proba(v)[:, 1], s])),
    }
    grid = np.arange(0.15, 0.81, 0.05)
    print(f"\n{'model':26s} {'thr':>5s}   dev R/P (F1)         test R/P (F1)        new-domain R/P (F1)")
    for label, (train_cols, test_cols) in variants.items():
        outs = []
        for seed in range(3):
            m = gbt(train_cols(), seed)
            fn = lambda x, v, s, m=m: m.predict_proba(test_cols(x, v, s))[:, 1]
            thr = max(grid, key=lambda t: evaluate("dev", fn, t)[2])
            outs.append((thr, {k: evaluate(k, fn, thr) for k in ("dev", "test", "unseen")}))
        cell = lambda k: "{:5.1%}/{:5.1%} ({:.3f})".format(*[np.mean([o[1][k][i] for o in outs]) for i in range(3)])
        print(f"{label:26s} {np.mean([o[0] for o in outs]):5.2f}   {cell('dev')}   {cell('test')}   {cell('unseen')}", flush=True)
    # 탐침 하나만 (트리 없이): 벡터가 얼마나 알려 주는가
    fn = lambda x, v, s: probe.predict_proba(v)[:, 1]
    thr = max(grid, key=lambda t: evaluate("dev", fn, t)[2])
    print(f"{'probe alone':26s} {thr:5.2f}   " + "   ".join("{:5.1%}/{:5.1%} ({:.3f})".format(*evaluate(k, fn, thr)) for k in ("dev", "test", "unseen")))


if __name__ == "__main__":
    main()
