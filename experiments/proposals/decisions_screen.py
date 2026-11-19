"""Haiku 선별을 OpenAI Decisions 로 바꿀 수 있는가: 쓸모 있는 쌍을 가려내는 정밀도·재현율·지연 비교 (프로토타입).

  uv run --env-file .env python experiments/proposals/decisions_screen.py [--limit 6]

운영 선별(`pair_screen.py`, Haiku)은 쌍 10개씩 묶음 8개를 동시에 물어 후보 60쌍을 4.6초에 거른다. Decisions 는 질문마다 확률을 주므로 임계값을 정할 수 있고
(Haiku 는 라벨 하나뿐이다), 한 요청에 질문 400개까지 담긴다. 세 가지를 시험한다 (문구는 한 번 정했고 라벨을 보고 고치지 않았다):
  choice  쌍마다 요청 하나, 객관식 질문 (선별과 같은 다섯 라벨) → P(useful)
  pred    쌍마다 요청 하나, 예/아니오 질문 → P(참)
  batch   60쌍을 한 문서로 묶어 요청 하나, 객관식 질문 60개 → P(useful) (운영 후보 60쌍과 같은 모양)
평가 쌍은 운영 후보와 같은 조건 (무작위 표본 제외, cos 0.5 이상). 라벨은 모델 라벨이다 (AI 라벨러 둘 중 하나라도 useful, 사람 표 150쌍 안에서는 Sonnet·Gemini·셋 중 둘도).
쌍의 두 문장이 OpenAI 로 나간다. 답은 data/private/cache/experiments/proposals/decisions_screen.pkl 에 캐시한다.
"""

import argparse
import csv
import hashlib
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.proposals.relation_features import CACHE  # noqa: E402

GOLD = ROOT / "data/private/gold"
STORE = CACHE / "decisions_screen.pkl"
MODEL = "gpt-6-luna"
BATCH = 60                  # 운영 후보 수 (기본 60)

TASK = ("Two statements A and B come from a student's lecture notes (two different notes, two different keywords). "
        "Decide whether combining A and B (plus ordinary well-established general knowledge) can produce a NEW piece of knowledge for the student: "
        "a specific, correct conclusion that is stated in neither statement alone and is non-trivial "
        "(e.g. a chain of reasoning A->B->C, a meaningful comparison or contrast, applying a general principle from one to the case in the other, one supplying a missing condition or step for the other). "
        "Be strict: if the only thing you can say is generic ('both concern X', 'X is related to Y'), it is not new knowledge.")
CHOICES = [{"value": "useful", "description": "combining them yields a specific, correct, non-trivial conclusion stated in neither statement alone"},
           {"value": "vague", "description": "related, but the only thing to say is generic; no specific conclusion"},
           {"value": "redundant", "description": "one statement already says what the other or the combination would conclude; nothing new"},
           {"value": "unrelated", "description": "no meaningful connection (shared word only, different senses, different subjects)"},
           {"value": "unusable", "description": "a statement is a fragment (bare heading, table cell, title, course-admin line) and cannot be judged"}]
PRED = TASK + " Answer TRUE only if such a specific, correct, new conclusion exists. FALSE for generic relatedness, redundancy, unrelated statements, or fragments."


def doc(p) -> str:
    return f"A [{p['A_keyword']}]: {p['A_text']}\nB [{p['B_keyword']}]: {p['B_text']}"


def key(variant: str, text: str) -> str:
    return hashlib.sha256(f"{MODEL}\x00{variant}\x00{TASK}\x00{text}".encode()).hexdigest()


def call(client, store, variant: str, text: str, questions: list[dict]):
    """요청 하나 → (답 목록, 지연 초). 캐시에 있으면 저장된 지연을 그대로 돌려준다."""
    k = key(variant, text + "\x00" + repr(questions))
    if k in store:
        return store[k]
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            d = client.decisions.create(model=MODEL, input=text, questions=questions)
            break
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(3 + 4 * attempt)
    out = []
    for a in d.answers:
        if a.type == "refusal":
            out.append(None)
        elif hasattr(a, "probabilities"):
            out.append({x.value: x.probability for x in a.probabilities})
        else:
            out.append(a.probability)
    store[k] = (out, time.perf_counter() - t0)
    return store[k]


def run_single(client, store, pairs, variant, workers):
    def one(p):
        q = ({"type": "choice", "instructions": TASK, "choices": CHOICES} if variant == "choice"
             else {"type": "predicate", "instructions": PRED})
        (ans,), secs = call(client, store, variant, doc(p), [q])
        return (0.0 if ans is None else (ans.get("useful", 0.0) if variant == "choice" else ans)), secs
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, pairs))
    return np.array([r[0] for r in res]), np.array([r[1] for r in res]), time.perf_counter() - t0


def run_batch(client, store, pairs, workers):
    chunks = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]

    def one(chunk):
        text = "\n\n".join(f"[pair {i}]\n{doc(p)}" for i, p in enumerate(chunk))
        qs = [{"type": "choice", "instructions": f"Judge [pair {i}] only. " + TASK, "choices": CHOICES} for i in range(len(chunk))]
        ans, secs = call(client, store, "batch", text, qs)
        return [0.0 if a is None else a.get("useful", 0.0) for a in ans], secs
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, chunks))
    return np.array([x for r in res for x in r[0]]), np.array([r[1] for r in res]), time.perf_counter() - t0


def auc(y, s):
    pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def load_labels(pairs):
    """평가 쌍마다 AI 라벨러 라벨, 사람 표 안의 쌍은 Sonnet·Gemini·셋 중 둘도 (표 밖은 None)."""
    rd = lambda f: {r["id"]: r for r in csv.DictReader(open(GOLD / f, encoding="utf-8"))}
    g, s, k = rd("pairing-human-gemini.csv"), rd("pairing-human-sonnet.csv"), rd("pairing-human-key.csv")
    src = {k[i]["source_id"]: i for i in g}
    out = {}
    for p in pairs:
        ai = "useful" in (p["label_a"], p["label_b"])
        row = {"AI 라벨러": ai}
        if p["id"] in src:
            i = src[p["id"]]
            so, ge = s[i]["label"] == "useful", g[i]["label"] == "useful"
            row.update({"Sonnet": so, "Gemini": ge, "셋 중 둘": (int(ai) + so + ge) >= 2})
        out[p["id"]] = row
    return out


def report(name, pairs, labels, haiku, score, secs, wall):
    n = len(pairs)
    hk = np.array([haiku[p["id"]] == "useful" for p in pairs])
    print(f"\n=== {name}: 평가 쌍 {n}개 · 요청 {len(secs)}번, 요청 지연 중앙값 {np.median(secs):.2f}s · 최대 {secs.max():.2f}s · 전체 {wall:.1f}s")
    print(f"선별(Haiku)이 고른 쌍 {int(hk.sum())}개. Decisions 점수 분포: 중앙값 {np.median(score):.2f} · 0.5 이상 {(score >= 0.5).sum()}개 · 0.3 이상 {(score >= 0.3).sum()}개")
    thr_n = int(hk.sum())
    top = np.zeros(n, bool)
    top[np.argsort(-score, kind="stable")[:thr_n]] = True            # Haiku 와 같은 개수를 고른다 (임계값을 라벨에 맞춰 고르지 않는다)
    print(f"Haiku 와 같은 개수({thr_n})를 위에서 고르면 두 방법이 겹친 쌍: {int((top & hk).sum())}개")
    print(f"\n{'라벨러':10s} {'쓸모':>4s} | {'Haiku 정밀/재현':>16s} | {'Decisions 위 '+str(thr_n):>18s} | {'≥0.5':>14s} | {'AUC':>5s}")
    for lab in ("AI 라벨러", "Sonnet", "Gemini", "셋 중 둘"):
        has = np.array([lab in labels[p["id"]] for p in pairs])
        if not has.any():
            continue
        y = np.array([labels[p["id"]].get(lab, False) for p in pairs])[has]
        h, t, th, s = hk[has], top[has], (score >= 0.5)[has], score[has]
        # 선별과 같은 개수 비교는 표 안의 쌍으로 다시 센다
        k = int(h.sum())
        tt = np.zeros(len(s), bool)
        tt[np.argsort(-s, kind="stable")[:k]] = True
        pr = lambda sel: f"{(sel & y).sum() / max(1, sel.sum()):4.0%} / {(sel & y).sum() / max(1, y.sum()):4.0%}"
        print(f"{lab:10s} {int(y.sum()):4d} | {pr(h):>16s} | {pr(tt):>18s} | {pr(th):>14s} | {auc(y, s):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", action="store_true", help="저장한 점수로 곡선·구간만 (호출 없음)")
    ap.add_argument("--limit", type=int, default=0, help="평가 쌍을 이만큼만 (시험 호출)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--variants", nargs="+", default=["choice", "pred", "batch"])
    args = ap.parse_args()
    import openai

    pairs = pickle.load(open(CACHE / "labeled.pkl", "rb"))["pairs"]
    haiku = {r["id"]: r["label"] for r in csv.DictReader(open(GOLD / "pairing-screen-claude-haiku-5-5.csv", encoding="utf-8"))}
    pairs = [p for p in pairs if p["arm"] != "random" and float(p["cos"] or 0) >= 0.5 and p["id"] in haiku]
    if args.limit:
        pairs = pairs[:args.limit]
    labels = load_labels(pairs)
    client = openai.OpenAI(max_retries=1, timeout=60)
    store = pickle.load(open(STORE, "rb")) if STORE.exists() else {}
    results = {}
    for v in args.variants:
        if v == "batch":
            sc, secs, wall = run_batch(client, store, pairs, args.workers)
        else:
            sc, secs, wall = run_single(client, store, pairs, v, args.workers)
        pickle.dump(store, open(STORE, "wb"))
        results[v] = sc
        report(v, pairs, labels, haiku, sc, secs, wall)
    pickle.dump({"ids": [p["id"] for p in pairs], "scores": results}, open(CACHE / "decisions_screen_scores.pkl", "wb"))




def curve():
    """저장한 점수로 (모델 호출 없음): 위 k 개 정밀도, Haiku 와 같은 개수에서의 차이의 부트스트랩 구간, 세 방법 평균, 선별 위에 겹치기."""
    d = pickle.load(open(CACHE / "decisions_screen_scores.pkl", "rb"))
    pairs = {p["id"]: p for p in pickle.load(open(CACHE / "labeled.pkl", "rb"))["pairs"]}
    ps = [pairs[i] for i in d["ids"]]
    haiku = {r["id"]: r["label"] for r in csv.DictReader(open(GOLD / "pairing-screen-claude-haiku-5-5.csv", encoding="utf-8"))}
    hk = np.array([haiku[i] == "useful" for i in d["ids"]])
    labels = load_labels(ps)
    from scipy.stats import rankdata
    S = dict(d["scores"])
    S["평균(세 방법)"] = np.mean([rankdata(v) for v in d["scores"].values()], axis=0) / len(hk)
    rng = np.random.default_rng(0)
    for lab in ("AI 라벨러", "Sonnet", "Gemini", "셋 중 둘"):
        has = np.array([lab in labels[i] for i in d["ids"]])
        y = np.array([labels[i].get(lab, False) for i in d["ids"]])[has]
        h = hk[has]
        k0 = int(h.sum())
        print(f"\n--- {lab} (평가 쌍 {has.sum()}개, 쓸모 {int(y.sum())}개, Haiku 가 고른 쌍 {k0}개: 정밀 {(h & y).sum() / max(1, k0):.0%} · 재현 {(h & y).sum() / max(1, y.sum()):.0%})")
        print(f"{'방법':14s} " + " ".join(f"위{k:>3d}:정밀/재현" for k in (10, 25, 40, 60)) + f" | 위{k0}−Haiku 차 [95% 구간]")
        for name, s in S.items():
            s = s[has]
            order = np.argsort(-s, kind="stable")
            cells = [f"{y[order[:k]].mean():4.0%}/{y[order[:k]].sum() / max(1, y.sum()):4.0%}" for k in (10, 25, 40, 60)]
            diffs = []
            for _ in range(2000):
                b = rng.integers(0, len(y), len(y))
                kb = int(h[b].sum())
                if kb == 0:
                    continue
                top = np.argsort(-s[b], kind="stable")[:kb]
                diffs.append(y[b][top].mean() - (h[b] & y[b]).sum() / kb)
            top0 = order[:k0]
            diff = y[top0].mean() - (h & y).sum() / max(1, k0)
            print(f"{name:14s} " + " ".join(f"{c:>14s}" for c in cells) + f" | {diff:+5.0%} [{np.percentile(diffs, 2.5):+.0%}, {np.percentile(diffs, 97.5):+.0%}]")
        # 선별(Haiku)이 useful 이라 한 쌍 안에서 Decisions 로 줄 세우기: 위 5 · 위 10
        sel = np.where(h)[0]
        for name in ("choice", "batch"):
            s = S[name][has]
            o = sel[np.argsort(-s[sel], kind="stable")]
            print(f"  Haiku useful {len(sel)}쌍을 {name} 점수로 줄 세우면 위 5: {y[o[:5]].sum()}/5 · 위 10: {y[o[:10]].sum()}/10 (선별 전체 {y[sel].sum()}/{len(sel)})")
        # 두 방법이 모두 고른 쌍 / 하나라도 고른 쌍 (Decisions 는 Haiku 와 같은 개수를 위에서)
        s = S["choice"][has]
        t = np.zeros(len(s), bool)
        t[np.argsort(-s, kind="stable")[:k0]] = True
        pr = lambda m: f"{int(m.sum()):3d}개 · 정밀 {(m & y).sum() / max(1, m.sum()):4.0%} · 재현 {(m & y).sum() / max(1, y.sum()):4.0%}"
        print(f"  Haiku ∩ choice 위{k0}: {pr(h & t)}   |   Haiku ∪ choice 위{k0}: {pr(h | t)}")


if __name__ == "__main__":
    curve() if "--curve" in sys.argv else main()
