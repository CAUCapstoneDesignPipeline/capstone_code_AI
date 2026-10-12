"""`uv run ai ...` 명령. ai/ 에서 실행한다.

  uv run ai serve [--host 127.0.0.1] [--port 8000] [--reload] [--dev]   # 내부 API 서버. --dev 면 /dev/ playground
  uv run ai concepts build-idf              # 기준 말뭉치(노트 전체)로 IDF 를 만든다
  uv run ai concepts extract                # 노트마다 개념 언급을 뽑아 mentions.jsonl 에 쓴다
  uv run ai concepts show <note_id>         # 노트 하나의 개념 요약
  uv run ai concepts eval --gold gold.csv   # 사람이 표시한 개념과 비교 (재현율·정밀도)
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from app.services.concepts import (ConceptExtractor, ExtractConfig, Idf, Mention, StaleIdfError, build_idf,
                                  concept_key, summarize)
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


def loose(s: str) -> str:
    return re.sub(r"[\s\-_·./()\[\]'\"]+", "", concept_key(s))


def cmd_eval(args):
    """gold.csv 열: note_id, concept. 같은 개념의 다른 표기는 | 로 잇는다 (WAL|Write Ahead Logging)."""
    gold: dict[str, list[set[str]]] = defaultdict(list)
    with args.gold.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            aliases = {loose(a) for a in nfc(row["concept"]).split("|") if loose(a)}
            if aliases:
                gold[nfc(row["note_id"]).strip()].append(aliases)
    extracted: dict[str, set[str]] = defaultdict(set)
    with args.mentions.open(encoding="utf-8") as f:
        for line in f:
            m = Mention.from_dict(json.loads(line))
            extracted[m.span.note_id].add(loose(m.key))

    tot_hit = tot_gold = tot_ext = tot_ext_hit = 0
    print(f"{'recall':>7} {'prec':>6}  note")
    for note_id in sorted(gold):
        ext = extracted.get(note_id, set())
        hit = sum(1 for aliases in gold[note_id] if aliases & ext)
        all_aliases = set().union(*gold[note_id])
        ext_hit = len(ext & all_aliases)
        tot_hit += hit; tot_gold += len(gold[note_id]); tot_ext += len(ext); tot_ext_hit += ext_hit
        print(f"{hit / len(gold[note_id]):7.0%} {ext_hit / len(ext) if ext else 0:6.0%}  {note_id[:50]}")
        if args.verbose:
            missed = [sorted(a)[0] for a in gold[note_id] if not a & ext]
            if missed:
                print("          놓침:", ", ".join(missed))
    print(f"\n전체 재현율 {tot_hit / tot_gold:.0%} ({tot_hit}/{tot_gold}) · "
          f"정밀도 {tot_ext_hit / tot_ext if tot_ext else 0:.0%} ({tot_ext_hit}/{tot_ext})")


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

    v = sub.add_parser("eval", help="사람이 표시한 개념과 비교")
    v.add_argument("--gold", type=Path, required=True)
    v.add_argument("--mentions", type=Path, default=OUT_DIR / "mentions.jsonl")
    v.add_argument("-v", "--verbose", action="store_true", help="놓친 개념도 출력")
    v.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    args.func(args)


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
    elif cmd == "serve":
        serve_main(rest)
    else:
        sys.exit(f"알 수 없는 명령: {cmd}")
