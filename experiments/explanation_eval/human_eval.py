"""⑤ 설명·검증의 사람 판정 일치율 (설계 7장: 사람 판정과 일치율 80% 이상).

  sample  노트마다 새 지식 제안을 만들어 사람이 표시할 표를 쓴다 (LLM 을 부른다: 설명 생성·검증·관계 판정)
  score   사람이 표시한 표로 시스템 판정과의 일치율을 잰다 (LLM 을 부르지 않는다)

  uv run python experiments/explanation_eval/human_eval.py sample --notes "OS-가상메모리" "DB-WAL" [--per-note 4]
  uv run python experiments/explanation_eval/human_eval.py score

파일 두 개 (data/private/gold/, 노트 내용이 들어 있어 저장소에 넣지 않는다)
  explanations.csv      사람이 표시할 표. 시스템 판정은 없다 (보고 표시하면 일치율이 부풀려진다). 줄 순서도 섞는다
    human   valid(근거에서 도출되고 새롭고 구체적이다) | invalid(틀렸거나, 이미 노트에 있거나, 뻔하다) | unsure
  explanations-key.csv  id → 시스템 판정. candidate(모든 검증 통과) | hypothesis(애매함이 남음) | rejected(설명 검증에서 제외)
시스템 판정을 사람 판정과 같은 세 값으로 본다: candidate → valid, hypothesis → unsure, rejected → invalid.
`uv run ai eval` 도 두 파일이 있으면 일치율을 신뢰 구간과 함께 보고서에 넣는다.
"""

import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SHEET = ROOT / "data/private/gold/explanations.csv"
FIELDS = ["id", "note", "linked_note", "source", "bridge", "target", "knowledge", "evidence", "human", "human_note"]
KEY_FIELDS = ["id", "system", "system_reasons"]
from app.services.evaluation import AS_HUMAN, LABELS  # noqa: E402


def key_path(sheet: Path) -> Path:
    return sheet.with_name(sheet.stem + "-key.csv")


def join(sheet_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    """표와 열쇠를 id 로 합쳐 score 가 받는 줄 (system, human)."""
    key = {r["id"]: r for r in key_rows}
    return [{**r, "system": key[r["id"]]["system"]} for r in sheet_rows]


def score(rows: list[dict]) -> dict:
    """사람이 표시한 줄만 센다. 일치율 = 시스템의 세 값과 사람 표시가 같은 비율.
    candidate 정밀도 = 시스템이 통과시킨 제안 중 사람이 valid 라고 한 비율 (사용자에게 바로 보이는 것의 품질).
    거른 것의 정확도 = 시스템이 rejected 한 제안 중 사람이 invalid 라고 한 비율."""
    done = [r for r in rows if (r.get("human") or "").strip() in LABELS]
    pairs = [(AS_HUMAN[r["system"]], r["human"].strip()) for r in done]
    confusion = Counter(pairs)
    cand = [h for r, (_, h) in zip(done, pairs) if r["system"] == "candidate"]
    rej = [h for r, (_, h) in zip(done, pairs) if r["system"] == "rejected"]
    return {"labelled": len(done), "total": len(rows),
            "agreement": sum(s == h for s, h in pairs) / len(pairs) if pairs else None,
            "candidate_precision": cand.count("valid") / len(cand) if cand else None,
            "rejected_accuracy": rej.count("invalid") / len(rej) if rej else None,
            "confusion": {f"{s}→{h}": n for (s, h), n in sorted(confusion.items())}}


def _rows(name: str, names: dict, res: dict, start: int) -> list[dict]:
    out = []
    for s in res["suggestions"]:
        quotes = [e["quote"] for st in s["steps"] for e in st.get("evidence", [])]
        out.append({"note": name, "linked_note": names.get(s["noteId"], s["noteId"]), "source": s["source"],
                    "bridge": s["bridge"], "target": s["target"], "system": s["status"],
                    "system_reasons": " / ".join(s["statusReasons"]), "knowledge": s["knowledge"],
                    "evidence": " | ".join(quotes)})
    for s in res["screened"]["rejected"]:
        out.append({"note": name, "linked_note": "", "source": s["source"], "bridge": s["bridge"], "target": s["target"],
                    "system": "rejected", "system_reasons": " / ".join(s["reasons"]),
                    "knowledge": s.get("knowledge", ""), "evidence": ""})
    for i, r in enumerate(out):
        r.update(id=f"x{start + i:04d}", human="", human_note="")
    return out


def blind(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(사람이 볼 표, 열쇠). 표는 id hash 순으로 섞어 같은 노트·같은 판정끼리 몰리지 않게 한다."""
    import hashlib
    sheet = sorted(({k: r[k] for k in FIELDS} for r in rows), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    return sheet, [{k: r[k] for k in KEY_FIELDS} for r in rows]


async def _sample(note_names: list[str], per_note: int) -> list[dict]:
    import httpx

    from app.client import AiClient, snapshot, suggest_knowledge, uid
    from app.core.config import Settings
    from app.dev.router import playground_policy
    from app.main import create_app

    settings = Settings.from_env()
    notes_dir = ROOT / "data/private/notes"
    import unicodedata
    texts = {unicodedata.normalize("NFC", p.stem): unicodedata.normalize("NFC", p.read_text(encoding="utf-8"))
             for p in sorted(notes_dir.glob("*.md"))}
    snaps = {n: snapshot(uid("note", n), n, t) for n, t in texts.items()}
    names = {s["noteId"]: n for n, s in snaps.items()}
    app = create_app(settings)
    rows = []
    async with app.router.lifespan_context(app):
        st = app.state
        if st.explainer is None or st.typer is None or st.edge_judge is None or st.explanation_judge is None:
            sys.exit("설명·관계·검증 LLM 이 모두 필요하다 (ANTHROPIC_API_KEY, AI_RELATION_LLM=1)")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ai", timeout=130) as http:
            client = AiClient("http://ai", settings.service_token, http,
                              concurrency=max(1, min(6, settings.max_inflight - 1)))
            for name in note_names:
                name = unicodedata.normalize("NFC", name)
                if name not in snaps:
                    sys.exit(f"노트를 찾지 못했다: {name}")
                res = await suggest_knowledge(client, snaps[name], [s for n, s in snaps.items() if n != name],
                                              playground_policy(), max_paths=per_note)
                rows += _rows(name, names, res, len(rows))
                print(f"{name}: 제안 {len(res['suggestions'])} · 제외 {len(res['screened']['rejected'])}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--notes", nargs="+", default=[])
    ap.add_argument("--per-note", type=int, default=4, help="노트마다 설명할 경로 수 (LLM 설명 호출 수)")
    ap.add_argument("--sheet", type=Path, default=SHEET)
    args = ap.parse_args()
    if args.cmd == "sample":
        if not args.notes:
            sys.exit("--notes 로 제안을 만들 노트를 고른다")
        if args.sheet.exists():
            sys.exit(f"{args.sheet} 가 이미 있다. 사람 표시를 덮어쓰지 않도록 다른 --sheet 를 준다")
        sheet, key = blind(asyncio.run(_sample(args.notes, args.per_note)))
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        for path, fields, rows in ((args.sheet, FIELDS, sheet), (key_path(args.sheet), KEY_FIELDS, key)):
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                w.writerows(rows)
        print(f"{len(sheet)}줄 → {args.sheet} (시스템 판정은 {key_path(args.sheet).name} 에 따로). "
              "열쇠는 열지 말고 human 칸에 valid / invalid / unsure 를 채운 뒤 score 를 돌린다")
        return
    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))
    s = score(join(read(args.sheet), read(key_path(args.sheet))))
    if not s["labelled"]:
        sys.exit("사람이 표시한 줄이 없다 (human 칸)")
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    print(f"표시 {s['labelled']} / {s['total']}줄 · 일치율 {pct(s['agreement'])} (목표 80%) · "
          f"candidate 정밀도 {pct(s['candidate_precision'])} · 제외 정확도 {pct(s['rejected_accuracy'])}")
    for k, n in s["confusion"].items():
        print(f"  시스템→사람 {k}: {n}")


if __name__ == "__main__":
    main()
