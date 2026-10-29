"""`uv run ai ...` 명령. ai/ 에서 실행한다.

  uv run ai serve [--host 127.0.0.1] [--port 8000] [--reload] [--dev]   # 내부 API 서버. --dev 면 /dev/ playground
  uv run ai concepts build-idf              # 기준 말뭉치(노트 전체)로 IDF 를 만든다
  uv run ai concepts extract                # 노트마다 개념 언급을 뽑아 mentions.jsonl 에 쓴다
  uv run ai concepts show <note_id>         # 노트 하나의 개념 요약
  uv run ai concepts eval [--include-draft]  # 정답 개념과 비교 (재현율·정밀도, docs/eval/gold-concepts.md)
  uv run ai links eval                      # ④ 노트 쌍 점수 오프라인 시험 (정답 쌍 Recall@k·MRR, Adamic-Adar × 새로움)
  uv run ai links refind                    # 숨겼다 다시 찾기 (정답 표시 없이, 노트를 숨기고 A-C 를 다시 찾는가)
  uv run ai derive [--auto] [--sound] [--show]   # 조건부 결론(R1·R2) 사례 시험 (docs/team/data). --auto 는 사실을 LLM 으로 뽑는다
  uv run ai eval [--include-draft]          # 위 평가 전부 + 설명 사람 판정을 한 번에. 신뢰 구간, dev/test, 지난 보고서와 비교
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

    from app.services.evaluation import assign_splits
    gold = load_gold(args.gold, include_draft=args.include_draft)
    if args.split != "all":                      # 규칙을 고칠 때는 dev 만 본다 (test 의 오답을 보면 test 가 test 가 아니게 된다)
        rows = list(csv.DictReader(args.gold.open(encoding="utf-8-sig", newline="")))
        splits = assign_splits(gold, {nfc(r["note_id"]).strip(): (r.get("split") or "").strip() for r in rows})
        gold = {n: g for n, g in gold.items() if splits[n] == args.split}
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
    v.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                   help="볼 정답 노트 (기본 dev: 규칙을 고칠 때 test 노트의 오답을 보지 않는다)")
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
    from app.services.links.evaluate import evaluate, hide_and_refind

    ap = argparse.ArgumentParser(prog="ai links")
    ap.add_argument("cmd", choices=["eval", "refind"])
    ap.add_argument("--notes", type=Path, default=NOTES_DIR)
    ap.add_argument("--idf", type=Path, default=OUT_DIR / "idf.json")
    ap.add_argument("--cache", type=Path, default=OUT_DIR / "cache")
    ap.add_argument("--pairs", type=Path, default=GOLD_DIR / "pairs.csv")
    ap.add_argument("--topics", type=Path, default=GOLD_DIR / "topics.csv")
    ap.add_argument("--json", type=Path, help="결과를 JSON 으로 저장")
    ap.add_argument("-v", "--verbose", action="store_true", help="정답 쌍마다 순위")
    ap.add_argument("--include-draft", action="store_true", help="eval: 검토 전(draft) 정답 쌍도 쓴다")
    ap.add_argument("--n", type=int, default=200, help="refind: 숨길 개념 쌍 수")
    ap.add_argument("--seed", type=int, default=0, help="refind: 쌍을 고르는 순서 (고정하면 같은 쌍)")
    args = ap.parse_args(argv)
    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    types = [(c, c, c) for c in ("is_a", "part_of", "requires", "causes", "used_for", "contrasts_with", "example_of")]
    ks = (1, 5, 10, 20)
    if args.cmd == "refind":
        texts = read_notes(args.notes)
        ex = load_extractor(args)
        corpus = build_corpus(texts, ex, lambda ms: select_concepts(ms, profile), types, profile.limits,
                              version=ex.method_version)
        results = hide_and_refind(corpus, n=args.n, seed=args.seed)
        print(f"노트 {len(texts)}개 · 숨긴 개념 쌍 {len(results[0].ranks)}개 (seed {args.seed}) · {ex.method_version}\n")
        print(f"{'방법':<22}" + "".join(f"{'R@' + str(k):>7}" for k in ks) + f"{'MRR':>7}")
        for r in results:
            print(f"{r.method:<22}" + "".join(f"{r.recall_at(k):7.0%}" for k in ks) + f"{r.mrr:7.2f}")
        if args.verbose:
            for (nid, a, c), rank in sorted(results[0].ranks.items(), key=lambda x: (x[1] or 10**9, x[0])):
                print(f"  {rank or '-':>4}  {a} → {c}  ({nid[:30]})")
        if args.json:
            args.json.write_text(json.dumps({
                "method": ex.method_version, "notes": len(texts), "pairs": len(results[0].ranks), "seed": args.seed,
                "results": [{"method": r.method, **{f"recall@{k}": round(r.recall_at(k), 4) for k in ks},
                             "mrr": round(r.mrr, 4)} for r in results]}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))
    from app.services.evaluation import gold_pairs
    pairs = gold_pairs(read(args.pairs), include_draft=args.include_draft)
    topics = {nfc(r["note_id"]).strip(): r["topic"] for r in read(args.topics)}
    texts = read_notes(args.notes)
    missing = sorted({n for a, b, _ in pairs for n in (a, b)} - set(texts)) + sorted(set(texts) - set(topics))
    if missing:
        sys.exit("노트·주제가 맞지 않는다:\n  " + "\n  ".join(missing))
    ex = load_extractor(args)
    corpus = build_corpus(texts, ex, lambda ms: select_concepts(ms, profile), types, profile.limits,
                          version=ex.method_version)
    rankings = evaluate(corpus, topics, [(a, b) for a, b, _ in pairs])

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


def eval_main(argv=None):
    """평가 묶음. 노트 색인은 한 번만 만든다. 보고서는 data/private/eval/ 에 쌓이고, 가장 최근 보고서와 비교한다."""
    import subprocess
    import time

    from app.api.v1.extract import select_concepts, to_concepts
    from app.core.config import EXTRACT_PROFILES
    from app.services.concepts.gold import Extracted, load_gold, loose, score
    from app.services.evaluation import (assign_splits, bootstrap, compare, gold_pairs, judged_precision,
                                        labelled_explanations, ratio)
    from app.services.links import build_corpus, related_notes
    from app.services.links.evaluate import evaluate, hide_and_refind

    ap = argparse.ArgumentParser(prog="ai eval")
    ap.add_argument("--notes", type=Path, default=NOTES_DIR)
    ap.add_argument("--idf", type=Path, default=OUT_DIR / "idf.json")
    ap.add_argument("--cache", type=Path, default=OUT_DIR / "cache")
    ap.add_argument("--gold", type=Path, default=GOLD_DIR / "concepts.csv")
    ap.add_argument("--pairs", type=Path, default=GOLD_DIR / "pairs.csv")
    ap.add_argument("--topics", type=Path, default=GOLD_DIR / "topics.csv")
    ap.add_argument("--judgments", type=Path, default=GOLD_DIR / "pair-judgments.csv",
                    help="판정한 노트 쌍 (query,candidate,label=yes|no,…,status)")
    ap.add_argument("--explanations", type=Path, default=GOLD_DIR / "explanations.csv")
    ap.add_argument("--include-draft", action="store_true", help="검토 전(draft) 정답 개념도 쓴다")
    ap.add_argument("--refind-n", type=int, default=200)
    ap.add_argument("--reports", type=Path, default=Path("data/private/eval"))
    ap.add_argument("--baseline", type=Path, help="비교할 보고서 (기본: 가장 최근 보고서)")
    ap.add_argument("--no-save", action="store_true", help="보고서를 쌓지 않는다 (규칙을 고치는 중 dev 만 볼 때)")
    ap.add_argument("--concept-filter", action="store_true", help="LLM 개념 거르기(extract-hybrid-3)를 켜고 잰다. LLM 을 부른다")
    ap.add_argument("--only", choices=["concepts"], help="개념 추출만 잰다")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="개념 추출을 잴 노트 (규칙·설정을 고치는 중에는 dev)")
    args = ap.parse_args(argv)

    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))
    texts = read_notes(args.notes)
    ex = load_extractor(args)
    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    metrics: dict[str, dict] = {}

    def put(name, est):
        metrics[name] = est.to_dict()
        print(f"  {name:<34} {est}")

    cfilter = None
    if args.concept_filter:                      # extract-hybrid-3 과 같이 LLM 이 개념 후보를 거른다 (노트마다 한 번, 캐시)
        from app.api.v1.extract import filter_concepts
        from app.main import role_llm_config
        from app.services.concepts.filter_llm import ConceptFilter
        from app.services.llm import make_llm
        cfg = role_llm_config("relation", "low")
        if cfg is None:
            sys.exit("--concept-filter 에는 LLM 설정이 필요하다 (ANTHROPIC_API_KEY 등)")
        cfilter = ConceptFilter(make_llm(cfg), Path("data/private/cache/relations/concepts"))
        keep_once = cfilter.keep

        def keep_retry(body, labels):            # 평가 스크립트만 일시 오류에 두 번 더 시도한다 (서버는 재시도 0, BE 가 재시도)
            from app.services.llm import LLMError
            for attempt in range(5):
                try:
                    return keep_once(body, labels)
                except LLMError as e:
                    if e.kind not in ("timeout", "provider", "rate_limit") or attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
        cfilter.keep = keep_retry

    def select(n):
        sel = select_concepts(ex.extract(n, texts[n]), profile)
        return filter_concepts(sel, texts[n], cfilter) if cfilter else sel

    # ① 개념 추출: 노트 단위 bootstrap, dev / test 따로
    gold = load_gold(args.gold, include_draft=args.include_draft)
    if not gold and not args.include_draft:      # 아직 검토한 정답이 없으면 초안으로 재고 그렇다고 적는다
        print("검토한(status=reviewed) 정답 개념이 없어 초안(draft)으로 잰다\n")
        args.include_draft = True
        gold = load_gold(args.gold, include_draft=True)
    splits = assign_splits(gold, {nfc(r["note_id"]).strip(): (r.get("split") or "").strip() for r in read(args.gold)})
    print(f"① 개념 추출 · 정답 노트 {len(gold)}개 (dev {sum(v == 'dev' for v in splits.values())} · "
          f"test {sum(v == 'test' for v in splits.values())}){' · draft 포함' if args.include_draft else ''}")
    per_note = {}
    for n in sorted(n for n in gold if args.split == "all" or splits.get(n) == args.split):
        if n not in texts:
            sys.exit(f"정답 노트가 노트 폴더에 없다: {n}")
        concepts, _ = to_concepts(select(n), texts[n])
        per_note[n] = score(gold[n], [Extracted(c.label, frozenset({loose(c.label), *(loose(m.quote) for m in c.mentions)}),
                                               c.confidence) for c in concepts])
    recall = ratio(lambda s: s.core_hit, lambda s: s.core)
    precision = ratio(lambda s: s.extracted_hit, lambda s: s.extracted)
    for part in (("dev", "test", "all") if args.split == "all" else (args.split,)):
        units = [s for n, s in sorted(per_note.items()) if part == "all" or splits.get(n) == part]
        put(f"concepts.recall_core.{part}", bootstrap(units, recall))
        put(f"concepts.precision.{part}", bootstrap(units, precision))

    if args.only == "concepts":
        return

    # ④ 노트 쌍과 숨겼다 다시 찾기: 같은 색인
    types = [(c, c, c) for c in ("is_a", "part_of", "requires", "causes", "used_for", "contrasts_with", "example_of")]
    def corpus_select(ms):
        sel = select_concepts(ms, profile)
        return filter_concepts(sel, texts[ms[0].span.note_id], cfilter) if cfilter and ms else sel
    corpus = build_corpus(texts, ex, corpus_select, types, profile.limits,
                          version=ex.method_version + ("+concept-filter" if cfilter else ""))
    pairs = [(a, b) for a, b, _ in gold_pairs(read(args.pairs), include_draft=args.include_draft)]
    topics = {nfc(r["note_id"]).strip(): r["topic"] for r in read(args.topics)}
    by = {r.method: r for r in evaluate(corpus, topics, pairs)}
    ranks = list(by["S (다른 주제)"].ranks.values())
    hit = lambda k: (lambda us: sum(1 for r in us if r is not None and r <= k) / len(us))
    print(f"④ 노트 쌍 · 정답 쌍 {len(pairs)}개{' (draft 포함)' if args.include_draft else ''} (질의 {len(ranks)}개, S 다른 주제)")
    put("links.recall@10", bootstrap(ranks, hit(10)))
    put("links.mrr", bootstrap(ranks, lambda us: sum(1 / r for r in us if r is not None) / len(us)))
    if args.judgments.exists():                  # 사용자에게 보이는 상위 3 개 중 이어진다고 판정된 비율
        judgments = [j for j in read(args.judgments)
                     if (j.get("status") or "reviewed").strip() == "reviewed" or args.include_draft]
        queries = sorted({nfc(j["query"]).strip() for j in judgments})
        top = {q: [l.note_b for l in related_notes(corpus, q, k=len(texts)) if topics.get(l.note_b) != topics.get(q)][:3]
               for q in queries if q in corpus.notes and topics.get(q) != "목차"}      # 목차 페이지는 링크 모음
        put("links.precision@3", bootstrap(judged_precision(top, judgments), lambda us: sum(us) / len(us)))
    refind = hide_and_refind(corpus, n=args.refind_n)
    print(f"④ 숨겼다 다시 찾기 · 쌍 {len(refind[0].ranks)}개")
    put("refind.recall@20", bootstrap(list(refind[0].ranks.values()), hit(20)))
    put("refind.baseline_recall@20", bootstrap(list(refind[1].ranks.values()), hit(20)))

    # ⑤ 설명 사람 판정 (표와 열쇠가 있고 표시한 줄이 있을 때)
    key = args.explanations.with_name(args.explanations.stem + "-key.csv")
    if args.explanations.exists() and key.exists():
        labelled = labelled_explanations(read(args.explanations), read(key))
        print(f"⑤ 설명 사람 판정 · 표시 {len(labelled)}줄")
        if labelled:
            put("explanations.agreement", bootstrap(labelled, lambda us: sum(s == h for s, h in us) / len(us)))
            cand = [h for s, h in labelled if s == "valid"]
            put("explanations.candidate_precision", bootstrap(cand, lambda us: us.count("valid") / len(us)))
    else:
        print("⑤ 설명 사람 판정 · 표 없음 (experiments/explanation_eval/human_eval.py sample)")

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "app"], capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "unknown", False
    import hashlib
    data = hashlib.sha256(b"".join(p.read_bytes() for p in (args.gold, args.pairs, args.topics, args.judgments)
                                   if p.exists())).hexdigest()[:12]
    report = {"commit": commit + ("+dirty" if dirty else ""), "data": data,
              "method": ex.method_version + ("+concept-filter" if cfilter else ""),
              "includeDraft": args.include_draft, "notes": len(texts), "metrics": metrics}
    old = sorted(args.reports.glob("*.json")) if args.reports.exists() else []
    base = args.baseline or (old[-1] if old else None)
    if base:
        prev = json.loads(base.read_text(encoding="utf-8"))
        if prev.get("includeDraft") != args.include_draft or prev.get("data") != data:
            print(f"\n(비교 생략: {base.name} 는 정답 자료가 다르다. 코드 변화와 자료 변화를 섞어 비교하지 않는다)")
        else:
            print(f"\n지난 보고서 {base.name} ({prev.get('commit')}) 와 비교 (지난 값의 95% 구간 기준)")
            for name, new, was, verdict in compare(metrics, prev["metrics"]):
                mark = {"나빠짐": "🔻", "좋아짐": "🔺"}.get(verdict, "  ")
                print(f"  {mark} {name:<34} {was:5.0%} → {new:5.0%}  {verdict}")
    if not args.no_save:
        args.reports.mkdir(parents=True, exist_ok=True)
        out = args.reports / f"{time.strftime('%Y%m%d-%H%M%S')}-{commit or 'unknown'}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n보고서 → {out}")


def derive_main(argv=None):
    """docs/team/data 의 사례를 돌려 기대와 비교한다. 기본은 사람이 적은 사실 + 규칙만 (LLM 없음, 같은 입력이면 같은 결과).
    --auto: 사실을 LLM 으로 뽑는다 (자동 추출 성능. 사람이 적은 사실과 따로 잰다), --sound: soundness 판정까지."""
    from app.main import role_llm_config
    from app.services.derive.cases import DATA, load_cases, run_case, score_case
    from app.services.llm import make_llm

    ap = argparse.ArgumentParser(prog="ai derive")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--case", nargs="*", help="이 사례만")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--auto", action="store_true", help="사실을 LLM 으로 뽑는다 (facts-llm-1)")
    ap.add_argument("--sound", action="store_true", help="규칙을 통과한 결론을 LLM 이 판정한다 (derive-sound-1)")
    ap.add_argument("--show", action="store_true", help="결론마다 설명을 보여 준다")
    ap.add_argument("--cache", type=Path, default=Path("data/private/cache"))
    args = ap.parse_args(argv)

    facts_of, judge, dropped = None, None, {}
    if args.auto:
        from app.services.derive.extract_llm import FactExtractor
        cfg = role_llm_config("facts", "medium")
        if cfg is None:
            sys.exit("사실 추출 LLM 이 없다 (ANTHROPIC_API_KEY 또는 AI_FACTS_LLM_PROVIDER)")
        ex = FactExtractor(make_llm(cfg), args.cache / "relations" / "facts")
        known: set[str] = set()

        def facts_of(n):
            got = ex.extract(n.note_id, 1, n.body, n.sha, sorted(known))
            known.update(f.subject_key for f in got.facts)
            for k, v in got.dropped.items():
                dropped[k] = dropped.get(k, 0) + v
            return got.facts
    if args.sound:
        from app.services.derive.sound_llm import SoundnessJudge
        cfg = role_llm_config("sound", "medium")
        if cfg is None:
            sys.exit("soundness LLM 이 없다 (ANTHROPIC_API_KEY 또는 AI_SOUND_LLM_PROVIDER)")
        judge = SoundnessJudge(make_llm(cfg), args.cache / "verifications" / "soundness")

    cases = [c for c in load_cases(args.data) if (not args.case or c["id"] in args.case)
             and args.split in ("all", c.get("split", "dev")) and (c.get("needs") != "sound" or judge is not None)]
    rows = []
    for c in cases:
        got, used = run_case(c, args.data, facts_of)
        if judge is not None:
            from app.services.derive.sound_llm import Premise, SoundIn, apply
            by_id = {f.fact_id: f for f in used}
            todo = [d for d in got if d.status == "conclusion"]
            if todo:
                verdicts, _, _ = judge.judge([SoundIn(d.derivation_id, d.conclusion, tuple(t for t, _ in d.conditions),
                                                      tuple(s.text for s in d.steps),
                                                      tuple(Premise(x, by_id[x].text(), by_id[x].evidence.quote)
                                                            for x in d.premises)) for d in todo])
                apply(got, {v.ref: v for v in verdicts}, by_id)
        s = score_case(c, got)
        rows.append(s)
        mark = "✓" if not s["mismatch"] else "✗"
        print(f"{mark} {c['id']:<32} 결론 {s['conclusions']} (맞음 {s['true_conclusions']}) · "
              f"기대 {s['expected']} 중 찾음 {s['found']} · 상태 {s['status_ok']}/{s['status_total']}  — {c.get('scene', '')}")
        for m in s["mismatch"]:
            print(f"    {m['key']}: 기대 {m['want']} · 결과 {m['got']}{' ' + m['result'] if m['result'] else ''}")
        if args.show:
            for d in got:
                print(f"    [{d.rule} {d.status}] {d.conclusion}")
                if d.status != "conclusion":
                    print("      " + " / ".join(d.reasons))
    tot = lambda k: sum(r[k] for r in rows)
    neg = [r for r in rows if r["negative"]]
    pct = lambda a, b: f"{a / b:.0%} ({a}/{b})" if b else "-"
    print(f"\n사례 {len(rows)}개 · 사실: {'LLM 추출' if args.auto else '사람이 적은 것'} · soundness: {'켬' if judge else '끔'}")
    print(f"  결론 정밀도      {pct(tot('true_conclusions'), tot('conclusions'))}")
    print(f"  결론 재현율      {pct(tot('found'), tot('expected'))}")
    print(f"  잘못된 확정      {pct(sum(r['false_confirm'] for r in neg), len(neg))} (음성 사례)")
    print(f"  상태 일치        {pct(tot('status_ok'), tot('status_total'))} (보류·기각·이미 앎 포함)")
    if dropped:
        print("  버린 사실        " + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))


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
    elif cmd == "eval":
        eval_main(rest)
    elif cmd == "derive":
        derive_main(rest)
    elif cmd == "serve":
        serve_main(rest)
    else:
        sys.exit(f"알 수 없는 명령: {cmd}")
