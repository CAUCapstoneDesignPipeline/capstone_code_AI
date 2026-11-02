"""`uv run ai ...` 명령. ai/ 에서 실행한다.

  uv run ai serve [--host 127.0.0.1] [--port 8000] [--reload] [--dev]   # 내부 API 서버. --dev 면 /dev/ playground
  uv run ai concepts build-idf              # 기준 말뭉치(노트 전체)로 IDF 를 만든다
  uv run ai concepts extract                # 노트마다 개념 언급을 뽑아 mentions.jsonl 에 쓴다
  uv run ai concepts show <note_id>         # 노트 하나의 개념 요약
  uv run ai concepts eval [--include-draft]  # 정답 개념과 비교 (재현율·정밀도, docs/eval/gold-concepts.md)
  uv run ai links eval                      # ④ 노트 쌍 점수 오프라인 시험 (정답 쌍 Recall@k·MRR, Adamic-Adar × 새로움)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from app.services.concepts import (ConceptExtractor, ExtractConfig, Idf, StaleIdfError, build_idf,
                                  summarize)
from app.core.config import EXTRACT_PROFILE
from app.services.concepts.markdown import nfc

NOTES_DIR = Path("data/private/notes")
OUT_DIR = Path("data/private/concepts")


def read_notes(notes_dir: Path, ids: list[str] | None = None) -> dict[str, str]:
    paths = {nfc(p.stem): p for p in sorted(notes_dir.glob("*")) if p.suffix in (".md", ".txt")}
    if ids:
        missing = [i for i in ids if i not in paths]
        if missing:
            sys.exit("노트를 찾지 못했다:\n  " + "\n  ".join(missing))
        paths = {i: paths[i] for i in ids}
    if not paths:
        sys.exit(f"노트가 없다: {notes_dir}")
    return {i: p.read_text(encoding="utf-8") for i, p in paths.items()}


def read_ids(path: Path | None) -> list[str] | None:
    if not path:
        return None
    return [nfc(l).strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def cmd_build_idf(args):
    texts = read_notes(args.notes)
    ex = ConceptExtractor(ExtractConfig())
    idf = build_idf(ex, texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    idf.save(args.out, ex.config_version)
    print(f"📚 IDF {idf.version}: 노트 {idf.n_docs}개 · 후보 {len(idf.df):,}개 → {args.out}")


def load_extractor(args) -> ConceptExtractor:
    if not args.idf.exists():
        sys.exit(f"IDF 가 없다: {args.idf}\n먼저 `ai concepts build-idf` 를 실행한다.")
    try:
        return ConceptExtractor(ExtractConfig(), Idf.load(args.idf), cache_dir=args.cache)
    except StaleIdfError as e:
        sys.exit(f"🛑 {e}")


def cmd_extract(args):
    texts = read_notes(args.notes, args.ids or read_ids(args.selected))
    ex = load_extractor(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_mentions = n_concepts = 0
    with args.out.open("w", encoding="utf-8") as f:
        for note_id in sorted(texts):
            ms = ex.extract(note_id, texts[note_id])
            for m in ms:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
            n_mentions += len(ms)
            n_concepts += len({m.key for m in ms})
    print(f"🧩 노트 {len(texts)}개 · 개념 {n_concepts:,}개 · 언급 {n_mentions:,}개 · {ex.method_version} → {args.out}")


def cmd_show(args):
    texts = read_notes(args.notes, [nfc(args.note_id)])
    ex = load_extractor(args)
    rows = summarize(ex.extract(args.note_id, texts[nfc(args.note_id)]))
    print(f"{'score':>6}  {'n':>3}  md  label")
    for r in rows[: args.top]:
        print(f"{r['score']:6.3f}  {r['mentions']:3d}  {'✓ ' if r['markdown'] else '  '}  {r['label']}")
    print(f"\n개념 {len(rows)}개 · {ex.method_version}")


THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def cmd_eval(args):
    """정답(docs/eval/gold-concepts.md 형식)과 /extract 와 같은 선택 규칙을 거친 추출 결과를 비교한다."""
    from app.api.v1.extract import select_concepts, to_concepts
    from app.core.config import EXTRACT_PROFILES
    from app.services.concepts.gold import Extracted, Score, load_gold, loose, score

    gold = load_gold(args.gold, include_draft=args.include_draft)
    if not gold:
        sys.exit("평가할 정답이 없다 (status=reviewed 인 줄이 없으면 --include-draft)")
    texts = read_notes(args.notes, sorted(gold))
    ex = load_extractor(args)
    profile = EXTRACT_PROFILES[args.profile]
    extracted = {}
    for note_id in sorted(texts):
        concepts, _ = to_concepts(select_concepts(ex.extract(note_id, texts[note_id]), profile), texts[note_id])
        extracted[note_id] = [Extracted(c.label, frozenset({loose(c.label), *(loose(m.quote) for m in c.mentions)}),
                                        c.confidence) for c in concepts]

    def at(t):
        return {n: [e for e in es if e.confidence >= t] for n, es in extracted.items()}

    print(f"정답 노트 {len(gold)}개 · core {sum(g.level == 'core' for gs in gold.values() for g in gs)}개 · "
          f"전체 {sum(len(gs) for gs in gold.values())}개 · {args.profile} · {ex.method_version}"
          f"{' · draft 포함' if args.include_draft else ''}\n")
    print(f"{'신뢰도≥':>7} {'재현율core':>10} {'재현율all':>9} {'정밀도':>6} {'추출 수':>6}")
    rows = []
    for t in THRESHOLDS:
        tot = Score()
        for n, es in at(t).items():
            tot.add(score(gold[n], es))
        rows.append({"threshold": t, "recall_core": round(tot.recall_core, 4), "recall_all": round(tot.recall_all, 4),
                     "precision": round(tot.precision, 4), "extracted": tot.extracted})
        print(f"{t:7.1f} {tot.recall_core:10.0%} {tot.recall_all:9.0%} {tot.precision:6.0%} {tot.extracted:6d}")

    print(f"\n노트별 (신뢰도 ≥ {args.min_confidence})")
    for n, es in sorted(at(args.min_confidence).items()):
        s = score(gold[n], es)
        print(f"  재현율 {s.recall_core:4.0%} · 정밀도 {s.precision:4.0%}  {n[:50]}")
        if args.verbose:
            if s.missed:
                print("      놓침:", ", ".join(s.missed))
            if s.wrong:
                print("      정답 밖:", ", ".join(s.wrong[:25]) + (" …" if len(s.wrong) > 25 else ""))
    if args.json:
        args.json.write_text(json.dumps({"profile": args.profile, "method": ex.method_version,
                                         "includeDraft": args.include_draft, "notes": len(gold), "thresholds": rows},
                                        ensure_ascii=False, indent=2), encoding="utf-8")


def concepts_main(argv=None):
    ap = argparse.ArgumentParser(prog="ai concepts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, idf=True):
        p.add_argument("--notes", type=Path, default=NOTES_DIR)
        if idf:
            p.add_argument("--idf", type=Path, default=OUT_DIR / "idf.json")
            p.add_argument("--cache", type=Path, default=OUT_DIR / "cache")

    b = sub.add_parser("build-idf", help="기준 말뭉치로 IDF 를 만든다")
    common(b, idf=False)
    b.add_argument("--out", type=Path, default=OUT_DIR / "idf.json")
    b.set_defaults(func=cmd_build_idf)

    e = sub.add_parser("extract", help="노트마다 개념 언급을 뽑는다")
    common(e)
    e.add_argument("--ids", nargs="*", help="일부 note id 만")
    e.add_argument("--selected", type=Path, help="note id 목록 파일 (한 줄에 하나)")
    e.add_argument("--out", type=Path, default=OUT_DIR / "mentions.jsonl")
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("show", help="노트 하나의 개념 요약")
    common(s)
    s.add_argument("note_id")
    s.add_argument("--top", type=int, default=40)
    s.set_defaults(func=cmd_show)

    v = sub.add_parser("eval", help="사람이 표시한 개념과 비교 (재현율·정밀도)")
    common(v)
    v.add_argument("--gold", type=Path, default=Path("data/private/gold/concepts.csv"))
    v.add_argument("--profile", default=EXTRACT_PROFILE)
    v.add_argument("--min-confidence", type=float, default=0.0, help="노트별 결과에 쓸 신뢰도 문턱값")
    v.add_argument("--include-draft", action="store_true", help="검토 전(draft) 정답도 쓴다")
    v.add_argument("--json", type=Path, help="문턱값별 결과를 JSON 으로 저장")
    v.add_argument("-v", "--verbose", action="store_true", help="놓친 개념·정답 밖 개념도 출력")
    v.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    args.func(args)


GOLD_DIR = Path("data/private/gold")


def links_main(argv=None):
    """④ 노트 쌍 점수 오프라인 시험. 정답 쌍(note_a,note_b,bridge_concept)과 노트 주제(note_id,topic)가 필요하다."""
    from app.api.v1.extract import select_concepts
    from app.core.config import EXTRACT_PROFILES
    from app.services.links import build_corpus
    from app.services.links.evaluate import evaluate

    ap = argparse.ArgumentParser(prog="ai links eval")
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--notes", type=Path, default=NOTES_DIR)
    ap.add_argument("--idf", type=Path, default=OUT_DIR / "idf.json")
    ap.add_argument("--cache", type=Path, default=OUT_DIR / "cache")
    ap.add_argument("--pairs", type=Path, default=GOLD_DIR / "pairs.csv")
    ap.add_argument("--topics", type=Path, default=GOLD_DIR / "topics.csv")
    ap.add_argument("--json", type=Path, help="결과를 JSON 으로 저장")
    ap.add_argument("-v", "--verbose", action="store_true", help="정답 쌍마다 순위")
    args = ap.parse_args(argv)

    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))
    pairs = [(nfc(r["note_a"]).strip(), nfc(r["note_b"]).strip(), r.get("bridge_concept", "")) for r in read(args.pairs)]
    topics = {nfc(r["note_id"]).strip(): r["topic"] for r in read(args.topics)}
    texts = read_notes(args.notes)
    missing = sorted({n for a, b, _ in pairs for n in (a, b)} - set(texts)) + sorted(set(texts) - set(topics))
    if missing:
        sys.exit("노트·주제가 맞지 않는다:\n  " + "\n  ".join(missing))
    ex = load_extractor(args)
    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    types = [(c, c, c) for c in ("is_a", "part_of", "requires", "causes", "used_for", "contrasts_with", "example_of")]
    corpus = build_corpus(texts, ex, lambda ms: select_concepts(ms, profile), types, profile.limits,
                          version=ex.method_version)
    rankings = evaluate(corpus, topics, [(a, b) for a, b, _ in pairs])

    ks = (1, 5, 10, 20)
    print(f"노트 {len(texts)}개 · 주제 {len(set(topics.values()))}개 · 정답 쌍 {len(pairs)}개 (질의 {2 * len(pairs)}개) · "
          f"{ex.method_version}\n")
    print(f"{'방법':<26}" + "".join(f"{'R@' + str(k):>7}" for k in ks) + f"{'MRR':>7}")
    for r in rankings:
        print(f"{r.method:<26}" + "".join(f"{r.recall_at(k):7.0%}" for k in ks) + f"{r.mrr:7.2f}")
    if args.verbose:
        print("\n정답 쌍 순위 (a 로 물었을 때 b, b 로 물었을 때 a)")
        for a, b, bridge in pairs:
            cells = "  ".join(f"{r.short}:{r.ranks[(a, b)] or '-'}/{r.ranks[(b, a)] or '-'}" for r in rankings)
            print(f"  {a[:24]} ↔ {b[:24]} ({bridge})\n      {cells}")
    if args.json:
        args.json.write_text(json.dumps({
            "method": ex.method_version, "notes": len(texts), "pairs": len(pairs),
            "results": [{"method": r.method, **{f"recall@{k}": round(r.recall_at(k), 4) for k in ks},
                         "mrr": round(r.mrr, 4)} for r in rankings]}, ensure_ascii=False, indent=2), encoding="utf-8")


def serve_main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(prog="ai serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--dev", action="store_true", help="개발용 playground (/dev/) 를 켠다. 로컬에서만")
    args = ap.parse_args(argv)
    if args.dev:
        import os
        os.environ["AI_DEV_UI"] = "1"
        os.environ.setdefault("AI_EMBEDDING_PROFILES", "embeddings-hash-dev-1")
        print(f"🧪 playground: http://{args.host}:{args.port}/dev/")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    """`uv run ai <명령> ...` 진입점."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "concepts":
        concepts_main(rest)
    elif cmd == "links":
        links_main(rest)
    elif cmd == "serve":
        serve_main(rest)
    else:
        sys.exit(f"알 수 없는 명령: {cmd}")
