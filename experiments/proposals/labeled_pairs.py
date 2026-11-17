"""라벨이 있는 후보 쌍 275개를 지금 색인의 요소(노트·개념 id)에 이어 붙인다.

시트(pairing-sheet.csv, pairing2-sheet.csv)에는 키워드와 문장뿐이라, playground 와 같은 방법으로 색인을 만들어 문장으로 요소를 찾는다.
라벨은 AI 라벨러 둘(a, b)이 눈가림으로 붙인 것이고 열쇠(pairing-key.csv, pairing2-key.csv)에는 어느 방법이 낸 쌍인지·코사인이 있다. 사람 라벨은 아직 없다.

  uv run --env-file .env python experiments/proposals/labeled_pairs.py        # data/private/cache/experiments/proposals/labeled.pkl 에 쓴다
"""

import csv
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GOLD = ROOT / "data/private/gold"
OUT = ROOT / "data/private/cache/experiments/proposals/labeled.pkl"
SETS = [("pairing", "1"), ("pairing2", "2")]
USEFUL = "useful"


def read(name: str) -> list[dict]:
    return list(csv.DictReader(open(GOLD / f"{name}.csv", encoding="utf-8")))


def load_labels() -> list[dict]:
    """시트 줄 + 라벨 둘 + 열쇠. id 는 'p1-3' 처럼 시트 번호를 붙인다."""
    out = []
    for stem, tag in SETS:
        sheet = {r["id"]: r for r in read(f"{stem}-sheet")}
        a = {r["id"]: r for r in read(f"{stem}-labels-a")}
        b = {r["id"]: r for r in read(f"{stem}-labels-b")}
        key = {r["id"]: r for r in read(f"{stem}-key")}
        for i, r in sheet.items():
            out.append({"id": f"s{tag}-{i}", "A_keyword": r["A_keyword"], "A_text": r["A_text"], "B_keyword": r["B_keyword"], "B_text": r["B_text"],
                        "label_a": a[i]["label"], "label_b": b[i]["label"], "arm": key[i]["arm"], "cos": float(key[i]["cos"]) if key[i]["cos"] else None})
    return out


def build_index():
    """playground 와 같은 설정으로 색인(요소 + 임베딩)을 만든다. 처음엔 임베딩에 1분쯤 걸릴 수 있다."""
    os.environ.setdefault("AI_SERVICE_TOKEN", "local-dev-token")
    os.environ.setdefault("AI_EMBEDDING_PROFILES", "embeddings-hash-dev-1")
    os.environ.setdefault("AI_CONCEPT_CACHE", "data/private/concepts/cache")
    os.environ["AI_DEV_UI"] = "1"
    notes_dir = Path(os.environ.get("AI_NOTES_DIR", "data/private/notes")).expanduser()
    os.environ["AI_NOTES_DIR"] = str(notes_dir)
    import run_playground
    os.environ["AI_CONCEPT_IDF"] = str(run_playground.ensure_idf(notes_dir))
    from fastapi.testclient import TestClient

    from app.dev import router as R
    from app.main import app
    with TestClient(app) as client:
        state = client.app.state
        corpus, _ = R._links_corpus(state)
        index, _, _ = R._pairing_index(state, corpus, 0.0)
        return corpus, index


def attach(rows: list[dict], index) -> tuple[list[dict], list[str]]:
    by = {}
    for e in index.el:
        by.setdefault((e.keyword, e.text), []).append(e)
        by.setdefault((None, e.text), []).append(e)
    out, miss = [], []
    for r in rows:
        ea = by.get((r["A_keyword"], r["A_text"])) or by.get((None, r["A_text"]))
        eb = by.get((r["B_keyword"], r["B_text"])) or by.get((None, r["B_text"]))
        if not ea or not eb:
            miss.append(r["id"])
            continue
        out.append({**r, "a": ea[0].idx, "b": eb[0].idx})
    return out, miss


def main():
    rows = load_labels()
    corpus, index = build_index()
    pairs, miss = attach(rows, index)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"pairs": pairs, "n_elements": len(index.el)}, open(OUT, "wb"))
    useful = sum(1 for p in pairs if USEFUL in (p["label_a"], p["label_b"]))
    print(f"시트 {len(rows)}쌍 중 요소를 찾은 것 {len(pairs)} (못 찾은 {len(miss)}: {miss[:8]}) · 요소 {len(index.el)}개 · 노트 {len(corpus.notes)}개 · useful(둘 중 하나) {useful}")


if __name__ == "__main__":
    main()
