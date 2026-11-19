"""후보 쌍의 사람 라벨: 표시할 표를 만들고 (sample), 표시한 표로 AI 라벨러·순위 방법을 다시 잰다 (score).

  uv run python experiments/proposals/human_pairs.py sample [--n 150] [--seed 0]
  uv run python experiments/proposals/human_pairs.py score

지금까지 후보 쌍의 '쓸모 있음' 은 모두 AI 라벨러 둘의 눈가림 판정이었다 (docs/knowledge-graph.md 의 선별·풀 깊이·순위 수치가 전부 여기서 나온다).
이 표는 그 쌍 가운데 150개를 사람이 같은 다섯 라벨로 다시 표시하게 해서, AI 판정이 사람과 얼마나 맞는지와 순위 방법의 AUC 를 사람 라벨로 다시 잰다.

고르는 법: 운영 후보와 같은 조건(무작위 표본 제외, 코사인 0.5 이상) 가운데 둘 다 unusable 이라 한 쌍(요소가 혼자 읽히지 않아 후보 단계에서 이미 걸러진다)을 빼고,
AI 라벨러 둘 중 하나라도 useful 이라 한 쌍은 모두, 나머지는 무작위로 채운다. 줄 순서를 섞는다.

파일 (data/private/gold/, 노트 문장이 들어 있어 저장소에 넣지 않는다)
  pairing-human.csv       사람이 표시할 표: id, A_keyword, A_text, B_keyword, B_text, human, human_conclusion, human_note
  pairing-human-key.csv   id → 원래 쌍 id, AI 라벨 둘, 방법(arm), 코사인. 표시가 끝나기 전에는 열지 않는다 (보고 표시하면 일치율이 부풀려진다)
  pairing-human-guide.md  표시 안내 (라벨 뜻)
"""

import argparse
import csv
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GOLD = ROOT / "data/private/gold"
SHEET, KEY, GUIDE = GOLD / "pairing-human.csv", GOLD / "pairing-human-key.csv", GOLD / "pairing-human-guide.md"
CACHE = ROOT / "data/private/cache/experiments/proposals"
LABELS = ["useful", "vague", "redundant", "unrelated", "unusable"]
FIELDS = ["id", "A_keyword", "A_text", "B_keyword", "B_text", "human", "human_conclusion", "human_note"]
KEY_FIELDS = ["id", "source_id", "label_a", "label_b", "arm", "cos"]
GUIDE_TEXT = """# 후보 쌍 표시 안내

`pairing-human.csv` 의 `human` 칸에 아래 다섯 값 중 하나를 적습니다. 줄마다 서로 다른 두 노트에서 뽑은 문장 A·B 입니다.
**A 와 B 두 문장과 일반 상식을 합치면, 둘 중 어느 쪽에도 없는 구체적이고 맞는 새 결론이 나오는가** 를 봅니다.

| 값 | 뜻 |
| --- | --- |
| `useful` | 합치면 둘 중 어느 쪽에도 없는 구체적이고 맞는 결론이 나온다 (A→B→C 식 추론, 의미 있는 비교·대조, 한쪽 원리를 다른 쪽에 적용). 가볍지 않아야 한다 |
| `vague` | 관련은 있지만 "둘 다 X 를 다룬다" 정도밖에 말할 수 없다. 구체적인 결론이 없다 |
| `redundant` | 한쪽이 다른 쪽(또는 합친 결론)을 이미 말한다. 새로운 것이 없다 |
| `unrelated` | 의미 있는 연결이 없다 (낱말만 같거나, 같은 낱말의 다른 뜻이거나, 주제가 다르다) |
| `unusable` | 문장이 조각이라 혼자 읽을 수 없다 (제목·표 칸·강의 안내 줄 등). 판단할 수 없다 |

- `useful` 이면 `human_conclusion` 에 새 결론을 한 문장으로 적어 주면 좋습니다 (적지 않아도 됩니다). 맞는지 자신이 없으면 `useful` 로 하지 마세요 — 엄격하게.
- 망설여지면 더 엄격한 쪽 (`useful` 보다 `vague`) 으로 표시하고 `human_note` 에 이유를 적습니다.
- 표시가 끝나기 전에는 `pairing-human-key.csv` 를 열지 마세요 (AI 가 뭐라고 했는지 보면 판단이 쏠립니다).
- 한 줄에 30초 안팎, 150줄이면 한 시간 남짓입니다. 나눠서 해도 되고, 빈 칸은 점수에서 빠집니다.
"""


def sample(n: int, seed: int):
    d = pickle.load(open(CACHE / "labeled.pkl", "rb"))["pairs"]
    pool = [p for p in d if p["arm"] != "random" and (p["cos"] or 0) >= 0.5 and not (p["label_a"] == "unusable" and p["label_b"] == "unusable")]
    useful = [p for p in pool if "useful" in (p["label_a"], p["label_b"])]
    rest = [p for p in pool if p not in useful]
    rng = random.Random(seed)
    chosen = useful + rng.sample(rest, max(0, min(len(rest), n - len(useful))))
    rng.shuffle(chosen)
    GOLD.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for i, p in enumerate(chosen, 1):
            w.writerow({"id": i, "A_keyword": p["A_keyword"], "A_text": p["A_text"], "B_keyword": p["B_keyword"], "B_text": p["B_text"],
                        "human": "", "human_conclusion": "", "human_note": ""})
    with open(KEY, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, KEY_FIELDS)
        w.writeheader()
        for i, p in enumerate(chosen, 1):
            w.writerow({"id": i, "source_id": p["id"], "label_a": p["label_a"], "label_b": p["label_b"], "arm": p["arm"], "cos": p["cos"]})
    GUIDE.write_text(GUIDE_TEXT, encoding="utf-8")
    print(f"표 {len(chosen)}줄 (AI 가 useful 이라 한 {len(useful)}개 + 나머지 {len(chosen) - len(useful)}개; 후보 풀 {len(pool)}쌍 중 나머지 {len(rest)}개에서 무작위) → {SHEET}")
    print(f"안내 {GUIDE} · 열쇠 {KEY} (표시가 끝나기 전에는 열지 않는다)")


def kappa(x: list[str], y: list[str]) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(sheet: list[dict], key: list[dict], pool_size: int | None = None) -> dict:
    """표시한 줄만 센다. 쓸모 있음의 정의는 두 가지: AI 둘 중 하나라도 useful (union) / 둘 다 (both).
    - ai_useful_precision: AI 가 useful 이라 한 쌍 가운데 사람도 useful 이라 한 비율 (선별이 거른 것의 질)
    - ai_miss: AI 가 useful 이 아니라 한 쌍 가운데 사람이 useful 이라 한 비율 (표본 안에서)
    - kappa_a / kappa_b: 사람과 각 AI 라벨러의 다섯 라벨 일치 (카파)
    표본은 AI-useful 을 모두 넣고 나머지는 무작위라서 전체 비율을 말하려면 가중해야 한다 (prevalence)."""
    k = {r["id"]: r for r in key}
    done = [(r, k[r["id"]]) for r in sheet if (r.get("human") or "").strip() in LABELS]
    h = [r["human"].strip() for r, _ in done]
    union = [("useful" in (kk["label_a"], kk["label_b"])) for _, kk in done]
    both = [(kk["label_a"] == kk["label_b"] == "useful") for _, kk in done]
    hu = [x == "useful" for x in h]
    au = [(r, x) for (r, kk), x, u in zip(done, hu, union) if u]
    an = [(r, x) for (r, kk), x, u in zip(done, hu, union) if not u]
    out = {"labelled": len(done), "total": len(sheet),
           "human_useful": sum(hu), "ai_useful": sum(union), "ai_useful_labelled": len(au),
           "ai_useful_precision": (sum(x for _, x in au) / len(au)) if au else None,
           "ai_miss": (sum(x for _, x in an) / len(an)) if an else None, "ai_not_useful_labelled": len(an),
           "kappa_a": kappa(h, [kk["label_a"] for _, kk in done]), "kappa_b": kappa(h, [kk["label_b"] for _, kk in done]),
           "human_confusion": dict(sorted(Counter(h).items()))}
    if pool_size and an and au:
        n_union = sum(1 for _, kk in ((r, kk) for r, kk in done) if "useful" in (kk["label_a"], kk["label_b"]))
        out["pool_prevalence"] = (sum(x for _, x in au) / len(au) * n_union + out["ai_miss"] * (pool_size - n_union)) / pool_size if n_union else None
    return out


def rank_check(sheet, key):
    """사람 라벨로 순위 특징의 AUC 를 다시 잰다 (features.pkl 이 있을 때)."""
    from experiments.proposals.relation_features import auc
    f = pickle.load(open(CACHE / "features.pkl", "rb"))
    where = {m["id"]: i for i, m in enumerate(f["meta"])}
    k = {r["id"]: r for r in key}
    rows = [(where[k[r["id"]]["source_id"]], r["human"].strip() == "useful") for r in sheet if (r.get("human") or "").strip() in LABELS]
    if not rows:
        return {}
    idx = np.array([i for i, _ in rows])
    y = np.array([u for _, u in rows], dtype=int)
    return {name: auc(y, f["X"][name][idx]) for name in ("cos", "token_jaccard", "direct", "shared_nb", "nb_links", "rel_in_b")} | {"n": len(y), "useful": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", type=Path, help="score: `human` 칸 대신 이 파일(id,label)의 라벨로 잰다 (예: AI 가 표시한 pairing-human-sonnet.csv). 사람 라벨이 아니다")
    args = ap.parse_args()
    if args.cmd == "sample":
        if SHEET.exists() and any((r.get("human") or "").strip() for r in csv.DictReader(open(SHEET, encoding="utf-8"))):
            sys.exit("이미 표시한 표가 있다. 덮어쓰지 않는다 (지우고 다시 만들려면 파일을 직접 옮겨 둔다)")
        return sample(args.n, args.seed)
    sheet = list(csv.DictReader(open(SHEET, encoding="utf-8")))
    key = list(csv.DictReader(open(KEY, encoding="utf-8")))
    if args.labels:
        other = {r["id"]: r["label"] for r in csv.DictReader(open(args.labels, encoding="utf-8"))}
        sheet = [{**r, "human": other.get(r["id"], "")} for r in sheet]
        print(f"※ {args.labels.name} 의 라벨로 쟀다 — 사람 라벨이 아니다 (표의 human 칸은 쓰지 않았다)\n")
    d = pickle.load(open(CACHE / "labeled.pkl", "rb"))["pairs"]
    pool = [p for p in d if p["arm"] != "random" and (p["cos"] or 0) >= 0.5 and not (p["label_a"] == "unusable" and p["label_b"] == "unusable")]
    res = score(sheet, key, len(pool))
    who = "이 라벨" if args.labels else "사람"
    print(f"표시한 줄 {res['labelled']}/{res['total']}")
    if not res["labelled"]:
        return
    print(f"{who}이 useful: {res['human_useful']}개 · 라벨 분포 {res['human_confusion']}")
    if res["ai_useful_precision"] is not None:
        print(f"AI 가 useful 이라 한 {res['ai_useful_labelled']}개 중 {who}도 useful: {res['ai_useful_precision']:.0%}")
    if res["ai_miss"] is not None:
        print(f"AI 가 useful 이 아니라 한 {res['ai_not_useful_labelled']}개 중 {who}이 useful: {res['ai_miss']:.0%}")
    if res.get("pool_prevalence") is not None:
        print(f"후보 풀 전체에서 {who} 기준 쓸모 있는 비율 (가중 추정): {res['pool_prevalence']:.1%}")
    print(f"카파 ({who} 대 AI a / b): {res['kappa_a']:.2f} / {res['kappa_b']:.2f}")
    rc = rank_check(sheet, key)
    if rc:
        print(f"\n{who} 라벨로 본 순위 특징 AUC (쌍 {rc['n']}개, 쓸모 있음 {rc['useful']}개):")
        for k, v in rc.items():
            if k not in ("n", "useful"):
                print(f"  {k:16s} {v:.3f}")


if __name__ == "__main__":
    main()
