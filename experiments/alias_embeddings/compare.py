"""별칭 병합 임베딩 모델 비교 (설계 concept-pipeline.md ② · 7장 "잘못 합침 5% 이하").

정답: data/private/gold/concepts.csv 의 같은 줄 표기(`|`)는 같은 개념. 노트가 달라도 표기가 겹치는 줄은 같은 개념으로 묶는다.
  양성 = 같은 개념의 서로 다른 표기 쌍, 음성 = 다른 개념의 표기 쌍 (모든 쌍. BE 가 전체 개념에서 cosine 으로 찾는 것과 같다)
입력 형식 두 가지: 표기만 / "표기: 처음 나온 문장" (설계: 표기 + 언급 문장).
모델마다 따로 프로세스를 띄워 메모리(최대 RSS)·적재 시간·CPU 추론 시간을 잰다. CPU 만 쓴다 (EC2 와 같게).

  uv run --extra embeddings python experiments/alias_embeddings/compare.py [--include-draft]
결과: data/private/gold/alias-embeddings.json (노트 원문 일부가 들어 있어 저장소에 넣지 않는다)
"""

import argparse
import itertools
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS = {   # 이름 → (Hugging Face id, 고정 revision, 앞붙임)
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "query: "),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3", "query: "),
    "bge-m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", ""),
}


def build_items(include_draft: bool):
    """(표기 목록, 표기마다 개념 번호, 표기마다 문맥 문장)."""
    from app.services.concepts.gold import load_gold, loose
    from app.services.concepts.markdown import mask, nfc
    from app.services.relations.sentences import sentence_spans

    gold = load_gold(ROOT / "data/private/gold/concepts.csv", include_draft=include_draft)
    raw = {}
    import csv
    for row in csv.DictReader((ROOT / "data/private/gold/concepts.csv").open(encoding="utf-8-sig")):
        raw.setdefault(nfc(row["note_id"]).strip(), []).append([n.strip() for n in nfc(row["concept"]).split("|")])
    # 노트를 넘어 표기가 겹치면 같은 개념 (union-find)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    surface_note = {}
    for note, rows in sorted(raw.items()):
        if note not in gold:
            continue
        for names in rows:
            keys = [loose(n) for n in names if loose(n)]
            for n in names:
                if loose(n):
                    surface_note.setdefault(loose(n), (n, note))
            for k in keys[1:]:
                parent[find(keys[0])] = find(k)
            find(keys[0])
    keys = sorted(surface_note)
    texts = {}
    for k in keys:
        name, note = surface_note[k]
        body = nfc((ROOT / "data/private/notes" / f"{note}.md").read_text(encoding="utf-8"))
        masked = mask(body)
        sent = next((body[s:e] for s, e in sentence_spans(body) if loose(name) in loose(masked[s:e])), "")
        texts[k] = (name, sent[:300])
    return keys, [find(k) for k in keys], texts


def measure(model: str, items: list[str]) -> dict:
    """자식 프로세스: 모델 하나 적재·추론. 벡터와 자원 사용량을 돌려준다."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(int(os.environ.get("EMB_THREADS", "4")))
    repo, rev, prefix = MODELS[model]
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    st = SentenceTransformer(repo, revision=rev, device="cpu", local_files_only=True)
    load_s = time.perf_counter() - t0
    loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    st.encode([prefix + "워밍업"], normalize_embeddings=True)
    t0 = time.perf_counter()
    vecs = st.encode([prefix + t for t in items], batch_size=32, normalize_embeddings=True)
    batch_s = time.perf_counter() - t0
    singles = []
    for t in items[:20]:
        t1 = time.perf_counter()
        st.encode([prefix + t], normalize_embeddings=True)
        singles.append(time.perf_counter() - t1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024          # macOS 는 바이트, Linux 는 KB
    return {"dims": int(vecs.shape[1]), "load_s": round(load_s, 2),
            "rss_loaded_mb": round((loaded - base) * scale / 1e6), "rss_peak_mb": round(peak * scale / 1e6),
            "batch_ms_per_item": round(1000 * batch_s / len(items), 2),
            "single_ms_p50": round(1000 * sorted(singles)[len(singles) // 2], 1),
            "torch_threads": torch.get_num_threads(), "vectors": vecs.round(6).tolist()}


def evaluate(vectors, groups) -> dict:
    """모든 표기 쌍의 cosine 으로 문턱값마다 병합 정밀도·재현율. 잘못 합침 5% 이하 중 재현율 최대 문턱값과 F1 최대 문턱값."""
    import numpy as np
    v = np.array(vectors)
    sims = v @ v.T
    pos, scores = [], []
    for i, j in itertools.combinations(range(len(groups)), 2):
        scores.append(float(sims[i, j]))
        pos.append(groups[i] == groups[j])
    scores, pos = np.array(scores), np.array(pos)
    n_pos = int(pos.sum())
    best_f1, best_safe, table = None, None, []
    for t in np.round(np.arange(0.50, 0.995, 0.01), 2):
        pred = scores >= t
        tp = int((pred & pos).sum())
        fp = int((pred & ~pos).sum())
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / n_pos if n_pos else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        row = {"threshold": float(t), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
               "merges": tp + fp, "false_merges": fp}
        table.append(row)
        if best_f1 is None or f1 > best_f1["f1"]:
            best_f1 = row
        if 1 - p <= 0.05 and tp + fp > 0 and (best_safe is None or r > best_safe["recall"]):
            best_safe = row
    # 순위 지표: 양성 쌍마다 그 표기의 최근접 이웃이 같은 개념인가 (BE 의 최근접 검색과 같은 관점)
    np.fill_diagonal(sims, -1)
    has_pair = [i for i in range(len(groups)) if sum(g == groups[i] for g in groups) > 1]
    nn_hit = sum(groups[int(sims[i].argmax())] == groups[i] for i in has_pair)
    return {"pairs": int(len(pos)), "positive_pairs": n_pos, "best_f1": best_f1, "best_safe": best_safe,
            "nearest_neighbor_acc": round(nn_hit / len(has_pair), 4) if has_pair else None, "table": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-draft", action="store_true")
    ap.add_argument("--child", nargs=2, metavar=("MODEL", "ITEMS_JSON"))
    args = ap.parse_args()
    if args.child:
        model, path = args.child
        print(json.dumps(measure(model, json.loads(Path(path).read_text(encoding="utf-8")))))
        return

    keys, groups, texts = build_items(args.include_draft)
    formats = {"label": [texts[k][0] for k in keys],
               "label+sentence": [f"{texts[k][0]}: {texts[k][1]}" if texts[k][1] else texts[k][0] for k in keys]}
    out = {"surfaces": len(keys), "concepts": len(set(groups)), "includeDraft": args.include_draft, "models": {}}
    tmp = ROOT / "data/private/gold/.alias-items.json"
    for model in MODELS:
        out["models"][model] = {}
        for fmt, items in formats.items():
            tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            res = subprocess.run([sys.executable, __file__, "--child", model, str(tmp)], capture_output=True, text=True,
                                 env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            if res.returncode:
                sys.exit(res.stderr[-2000:])
            m = json.loads(res.stdout.strip().splitlines()[-1])
            ev = evaluate(m.pop("vectors"), groups)
            out["models"][model][fmt] = {**m, **{k: v for k, v in ev.items() if k != "table"}, "table": ev["table"]}
            b, s = ev["best_f1"], ev["best_safe"] or {}
            print(f"{model:<22} {fmt:<15} dims {m['dims']:>4} · 메모리 {m['rss_loaded_mb']:>5} MB · 적재 {m['load_s']:>5}s · "
                  f"{m['batch_ms_per_item']:>6} ms/표기(묶음) · {m['single_ms_p50']:>6} ms(1개) | "
                  f"F1최대 τ={b['threshold']} P {b['precision']:.0%} R {b['recall']:.0%} | "
                  f"잘못합침≤5% τ={s.get('threshold', '-')} R {s.get('recall', 0):.0%} | 최근접 {ev['nearest_neighbor_acc']:.0%}",
                  flush=True)
    tmp.unlink(missing_ok=True)
    (ROOT / "data/private/gold/alias-embeddings.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")
    print(f"\n표기 {out['surfaces']}개 · 개념 {out['concepts']}개 · 양성 쌍 {ev['positive_pairs']} · 전체 쌍 {ev['pairs']}")


if __name__ == "__main__":
    main()
