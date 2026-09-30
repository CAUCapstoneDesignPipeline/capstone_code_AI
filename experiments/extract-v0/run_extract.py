"""추출 시험 v0 실행 스크립트 (README.md 절차 3~6).

  python run_extract.py extract   # 3. 추출 실행: 노트마다 N번, 원문 응답·파싱·시간·비용 기록
  python run_extract.py sheets    # 4~5. 채점표 relations.csv, merges.csv, pairs.csv 생성
  python run_extract.py metrics   # 6. 두 채점자 결과로 지표 표와 다음 주 판단 출력

입력 (커밋 금지, ai/data/private/extract-v0/ 아래):
  notes/*.md|*.txt   이름·연락처를 지운 노트. 파일 이름(확장자 제외)이 note_id
  gold_pairs.csv     추출 전에 적은 정답 연결 목록. 열: note_a,note_b,bridge_concept
원문 응답과 실행 기록은 노트 문장을 담으므로 ai/data/private/extract-v0/runs/ 에 둔다.
"""

import argparse
import csv
import difflib
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AI_ROOT = HERE.parents[1]
PRIVATE = AI_ROOT / "data" / "private" / "extract-v0"
NOTES_DIR = PRIVATE / "notes"
GOLD_PAIRS = PRIVATE / "gold_pairs.csv"
RUNS_DIR = PRIVATE / "runs"
RELATION_TYPES = HERE / "relation_types.json"

PROMPT_VERSION = "v0"
DEFAULT_MODEL = "claude-opus-5-5"
# USD per 1M tokens (input, output). 캐시 미사용 기준
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# 채점 라벨 (README "채점")
REL_LABELS = ["맞음", "유형 틀림", "방향 틀림", "근거 불일치", "없는 관계", "놓친 관계"]
EVI_LABELS = ["일치", "변형", "없음"]
MERGE_LABELS = ["맞음", "잘못 합침", "못 합침"]
PAIR_LABELS = ["양쪽 명시", "양쪽 명시지만 미병합", "한쪽만", "둘 다 없음"]

# 판단 기준 (README "핵심 지표", 수요일 확정 후 변경 금지)
THRESHOLDS = {
    "precision": (0.70, 0.50),
    "hallucination": 0.10,
    "evidence_match": 0.90,
    "wrong_merge": 0.10,
    "bridge_rate": (0.50, 0.20),
}


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def load_relation_types():
    return json.loads(RELATION_TYPES.read_text(encoding="utf-8"))["types"]


def load_notes():
    paths = sorted(p for p in NOTES_DIR.glob("*") if p.suffix in (".md", ".txt"))
    return {p.stem: nfc(p.read_text(encoding="utf-8")) for p in paths}


def load_gold_pairs():
    with GOLD_PAIRS.open(encoding="utf-8-sig", newline="") as f:
        return [{k: nfc(v).strip() for k, v in row.items()} for row in csv.DictReader(f)]


# ---------- 3. 추출 ----------

def build_system_prompt(types):
    type_lines = "\n".join(f"- {t['name']}: {t['description']}" for t in types)
    return f"""너는 공부 노트에서 개념 사이의 관계를 뽑는다.

규칙:
- 노트에 실제로 쓰여 있는 관계만 뽑는다. 상식이나 추론으로 관계를 만들지 않는다.
- 관계 유형은 아래 목록 중 하나만 쓴다.
{type_lines}
- 주어와 목적어는 노트에 나온 표기 그대로의 짧은 개념 이름으로 쓴다.
- evidence 는 그 관계의 근거가 되는 노트 문장을 한 글자도 바꾸지 않고 그대로 옮긴다.
- certainty 는 노트가 그 관계를 얼마나 분명히 말하는지 0.0~1.0 으로 적는다.

출력은 JSON 하나만, 다른 글 없이:
{{"relations": [{{"subject": "...", "relation_type": "...", "object": "...", "evidence": "...", "certainty": 0.0}}]}}"""


def parse_response(text, type_names):
    """원문 응답 → (관계 목록, 오류 메시지). 오류가 있으면 관계 목록은 비어 있다."""
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return [], f"json: {e}"
    rels = data.get("relations") if isinstance(data, dict) else None
    if not isinstance(rels, list):
        return [], "missing 'relations' list"
    keys = ("subject", "relation_type", "object", "evidence", "certainty")
    for i, r in enumerate(rels):
        if not isinstance(r, dict) or any(k not in r for k in keys):
            return [], f"relation {i}: missing keys"
        if r["relation_type"] not in type_names:
            return [], f"relation {i}: unknown type {r['relation_type']!r}"
    return rels, ""


def call_model(client, model, effort, system, note_text):
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": f"<note>\n{note_text}\n</note>"}],
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return response, text


def cmd_extract(args):
    import anthropic

    if not GOLD_PAIRS.exists():
        sys.exit(f"정답 연결 목록이 없다: {GOLD_PAIRS}\n추출을 돌리기 전에 먼저 적어 둔다 (README 데이터 3번).")
    notes = load_notes()
    if not notes:
        sys.exit(f"노트가 없다: {NOTES_DIR}")
    if args.notes:
        notes = {k: v for k, v in notes.items() if k in set(args.notes)}

    types = load_relation_types()
    type_names = {t["name"] for t in types}
    system = build_system_prompt(types)
    price_in, price_out = PRICES.get(args.model, (float("nan"), float("nan")))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "model": args.model, "effort": args.effort,
        "prompt_version": PROMPT_VERSION, "repeats": args.repeats,
        "relation_types": types, "system_prompt": system,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    client = anthropic.Anthropic()
    log_fields = ["run_id", "note_id", "repeat", "model_requested", "model_served", "prompt_version",
                  "effort", "elapsed_s", "input_tokens", "output_tokens", "cost_usd", "stop_reason",
                  "parse_ok", "parse_error", "n_relations", "api_error"]
    log_path = run_dir / "log.csv"
    parsed_path = run_dir / "parsed.jsonl"
    total_cost = 0.0

    with log_path.open("w", encoding="utf-8", newline="") as lf, parsed_path.open("w", encoding="utf-8") as pf:
        log = csv.DictWriter(lf, fieldnames=log_fields)
        log.writeheader()
        for note_id, text in notes.items():
            for rep in range(1, args.repeats + 1):
                row = {"run_id": run_id, "note_id": note_id, "repeat": rep, "model_requested": args.model,
                       "prompt_version": PROMPT_VERSION, "effort": args.effort}
                t0 = time.monotonic()
                try:
                    response, raw_text = call_model(client, args.model, args.effort, system, text)
                except anthropic.APIConnectionError as e:
                    row.update(elapsed_s=round(time.monotonic() - t0, 2), parse_ok=False, api_error=f"connection: {e}")
                    log.writerow(row); lf.flush()
                    print(f"[{note_id} #{rep}] 연결 오류: {e}")
                    continue
                except anthropic.APIStatusError as e:
                    row.update(elapsed_s=round(time.monotonic() - t0, 2), parse_ok=False,
                               api_error=f"{e.status_code}: {e.message}")
                    log.writerow(row); lf.flush()
                    print(f"[{note_id} #{rep}] API 오류 {e.status_code}: {e.message}")
                    continue
                elapsed = time.monotonic() - t0

                (run_dir / "raw" / f"{note_id}__r{rep}.json").write_text(
                    response.model_dump_json(indent=2), encoding="utf-8")

                if response.stop_reason in ("refusal", "max_tokens"):
                    rels, err = [], f"stop_reason={response.stop_reason}"
                else:
                    rels, err = parse_response(raw_text, type_names)
                u = response.usage
                cost = (u.input_tokens * price_in + u.output_tokens * price_out) / 1e6
                total_cost += cost
                row.update(model_served=response.model, elapsed_s=round(elapsed, 2),
                           input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                           cost_usd=round(cost, 5), stop_reason=response.stop_reason,
                           parse_ok=not err, parse_error=err, n_relations=len(rels), api_error="")
                log.writerow(row); lf.flush()
                pf.write(json.dumps({"note_id": note_id, "repeat": rep, "raw_text": raw_text,
                                     "parse_ok": not err, "relations": rels}, ensure_ascii=False) + "\n")
                pf.flush()
                print(f"[{note_id} #{rep}] {elapsed:.1f}s ${cost:.4f} "
                      f"{'OK ' + str(len(rels)) + '개' if not err else 'FAIL ' + err}")

    write_stability(run_dir)
    print(f"\n실행 {run_id} 완료. 총 비용 ${total_cost:.4f}\n기록: {run_dir}")


def triple(r):
    return (nfc(r["subject"]).strip().lower(), r["relation_type"], nfc(r["object"]).strip().lower())


def read_parsed(run_dir):
    with (run_dir / "parsed.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_stability(run_dir):
    """같은 노트의 반복 실행끼리 (주어, 유형, 목적어) 집합의 Jaccard 유사도."""
    by_note = defaultdict(dict)
    for rec in read_parsed(run_dir):
        if rec["parse_ok"]:
            by_note[rec["note_id"]][rec["repeat"]] = {triple(r) for r in rec["relations"]}
    rows = []
    for note_id, reps in sorted(by_note.items()):
        if 1 in reps and 2 in reps:
            a, b = reps[1], reps[2]
            union = a | b
            rows.append({"note_id": note_id, "n_r1": len(a), "n_r2": len(b), "n_common": len(a & b),
                         "jaccard": round(len(a & b) / len(union), 3) if union else 1.0})
    with (run_dir / "stability.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["note_id", "n_r1", "n_r2", "n_common", "jaccard"])
        w.writeheader()
        w.writerows(rows)
    if rows:
        mean = sum(r["jaccard"] for r in rows) / len(rows)
        print(f"반복 안정성 (Jaccard 평균, 노트 {len(rows)}개): {mean:.2f}")


# ---------- 4~5. 채점표 ----------

def latest_run_dir(run_id=None):
    if run_id:
        return RUNS_DIR / run_id
    runs = sorted(p for p in RUNS_DIR.glob("*") if (p / "parsed.jsonl").exists())
    if not runs:
        sys.exit(f"실행 기록이 없다: {RUNS_DIR}")
    return runs[-1]


def squash(s):
    return re.sub(r"\s+", " ", nfc(s)).strip()


def auto_evidence(evidence, note_text):
    """근거 문장 자동 판정 (채점 참고용): 일치 / 변형 / 없음."""
    ev = nfc(evidence)
    if ev and ev in note_text:
        return "일치", 1.0
    ev_s = squash(evidence)
    if not ev_s:
        return "없음", 0.0
    # 노트를 문장·줄 단위로 나눠 가장 비슷한 조각과 비교
    pieces = [squash(p) for p in re.split(r"(?<=[.!?。])\s+|\n+", note_text) if p.strip()]
    best = max((difflib.SequenceMatcher(None, ev_s, p).ratio() for p in pieces), default=0.0)
    return ("변형" if best >= 0.6 else "없음"), round(best, 2)


def concept_key(name):
    """병합 후보를 묶는 느슨한 키: 괄호 내용·공백·기호 제거, 소문자."""
    s = nfc(name).lower()
    s = re.sub(r"\(.*?\)", "", s)
    return re.sub(r"[\s\-_·.,/]+", "", s)


def write_csv(path, fields, rows, force):
    if path.exists() and not force:
        print(f"건너뜀 (이미 있음, 덮어쓰려면 --force): {path.name}")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"작성: {path.name} ({len(rows)}행)")


def cmd_sheets(args):
    run_dir = latest_run_dir(args.run)
    run_id = run_dir.name
    notes = load_notes()
    records = [r for r in read_parsed(run_dir) if r["repeat"] == args.repeat and r["parse_ok"]]

    # relations.csv
    rel_rows = []
    concepts = defaultdict(set)  # 표기 → note_id 집합
    for rec in records:
        text = notes.get(rec["note_id"], "")
        for i, r in enumerate(rec["relations"], 1):
            label, score = auto_evidence(r["evidence"], text)
            rel_rows.append({
                "run_id": run_id, "note_id": rec["note_id"], "rel_id": f"{rec['note_id']}-{i}",
                "subject": r["subject"], "relation_type": r["relation_type"], "object": r["object"],
                "evidence": r["evidence"], "certainty": r["certainty"],
                "auto_evidence": label, "auto_evidence_score": score,
                "rel_a": "", "rel_b": "", "rel_final": "",
                "evi_a": "", "evi_b": "", "evi_final": "", "memo": "",
            })
            for c in (r["subject"], r["object"]):
                concepts[nfc(c).strip()].add(rec["note_id"])
    write_csv(HERE / "relations.csv", list(rel_rows[0].keys()) if rel_rows else
              ["run_id", "note_id", "rel_id", "subject", "relation_type", "object", "evidence", "certainty",
               "auto_evidence", "auto_evidence_score", "rel_a", "rel_b", "rel_final",
               "evi_a", "evi_b", "evi_final", "memo"], rel_rows, args.force)

    # merges.csv: 같은 느슨한 키끼리 auto_group 으로 묶어 둔다. group 열에 최종 묶음을 적는다.
    groups = defaultdict(list)
    for surface in concepts:
        groups[concept_key(surface)].append(surface)
    merge_rows = []
    for key, surfaces in sorted(groups.items()):
        for s in sorted(surfaces):
            merge_rows.append({"surface": s, "note_ids": ";".join(sorted(concepts[s])),
                               "auto_group": key, "group": key,
                               "merge_a": "", "merge_b": "", "merge_final": "", "memo": ""})
    write_csv(HERE / "merges.csv",
              ["surface", "note_ids", "auto_group", "group", "merge_a", "merge_b", "merge_final", "memo"],
              merge_rows, args.force)

    # pairs.csv: 정답 쌍마다 매개 개념이 각 노트 원문·추출 결과에 있는지 참고값을 채운다.
    extracted = defaultdict(set)
    for rec in records:
        for r in rec["relations"]:
            extracted[rec["note_id"]].update({concept_key(r["subject"]), concept_key(r["object"])})
    pair_rows = []
    for i, p in enumerate(load_gold_pairs(), 1):
        bk = concept_key(p["bridge_concept"])
        row = {"pair_id": i, "note_a": p["note_a"], "note_b": p["note_b"], "bridge_concept": p["bridge_concept"]}
        for side in ("a", "b"):
            nid = p[f"note_{side}"]
            row[f"in_text_{side}"] = nfc(p["bridge_concept"]).lower() in notes.get(nid, "").lower()
            row[f"in_extract_{side}"] = any(bk and (bk in k or k in bk) for k in extracted.get(nid, ()) if k)
        row.update(pair_a="", pair_b="", pair_final="", memo="")
        pair_rows.append(row)
    write_csv(HERE / "pairs.csv",
              ["pair_id", "note_a", "note_b", "bridge_concept", "in_text_a", "in_extract_a",
               "in_text_b", "in_extract_b", "pair_a", "pair_b", "pair_final", "memo"],
              pair_rows, args.force)

    print("\n라벨 값:")
    print("  rel_*  :", " / ".join(REL_LABELS), "(놓친 관계는 새 행으로 추가)")
    print("  evi_*  :", " / ".join(EVI_LABELS))
    print("  merge_*:", " / ".join(MERGE_LABELS))
    print("  pair_* :", " / ".join(PAIR_LABELS))
    print("_a = 이건, _b = 팀원. 둘이 다를 때만 같이 보고 _final 을 적는다.")


# ---------- 6. 지표 ----------

def read_sheet(name):
    path = HERE / name
    if not path.exists():
        sys.exit(f"채점표가 없다: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def final_label(row, prefix):
    """_final 이 있으면 그것, 없으면 두 채점이 같을 때 그 값, 아니면 빈 문자열."""
    a, b, final = ((row.get(f"{prefix}_{s}") or "").strip() for s in ("a", "b", "final"))
    return final or (a if a and a == b else "")


def resolve(rows, prefix, allowed, sheet):
    """행마다 최종 라벨을 모은다. 미결(채점 없음·불일치) 행은 따로 센다."""
    out, pending, bad = [], [], []
    for row in rows:
        label = final_label(row, prefix)
        if not label:
            pending.append(row)
            continue
        if label not in allowed:
            bad.append((row, label))
            continue
        out.append(label)
    for row, label in bad:
        print(f"  [{sheet}] 알 수 없는 라벨 {label!r}: {row}")
    return out, pending


def ratio(n, d):
    return n / d if d else float("nan")


def cmd_metrics(args):
    rel_rows = read_sheet("relations.csv")
    rel, rel_pending = resolve(rel_rows, "rel", REL_LABELS, "relations")
    evi, evi_pending = resolve([r for r in rel_rows if final_label(r, "rel") != "놓친 관계"],
                               "evi", EVI_LABELS, "relations")
    merge, merge_pending = resolve(read_sheet("merges.csv"), "merge", MERGE_LABELS, "merges")
    pair, pair_pending = resolve(read_sheet("pairs.csv"), "pair", PAIR_LABELS, "pairs")

    judged = [l for l in rel if l != "놓친 관계"]
    missed = rel.count("놓친 관계")
    m = {
        "precision": ratio(judged.count("맞음"), len(judged)),
        "hallucination": ratio(judged.count("없는 관계"), len(judged)),
        "evidence_match": ratio(evi.count("일치"), len(evi)),
        "wrong_merge": ratio(merge.count("잘못 합침"), len(merge)),
        "bridge_rate": ratio(pair.count("양쪽 명시"), len(pair)),
    }

    def pct(x):
        return "  n/a" if x != x else f"{x * 100:5.1f}%"

    print("| 지표 | 값 | 표본 | 판단 |")
    print("| --- | --- | --- | --- |")
    hi, lo = THRESHOLDS["precision"]
    p = m["precision"]
    print(f"| 관계 정밀도 | {pct(p)} | {len(judged)} | "
          + ("v0로 2주차 파이프라인 연결" if p >= hi else "추출 방식 재검토" if p < lo else "보류 (50~70%)") + " |")
    h = m["hallucination"]
    print(f"| 없는 관계 비율 | {pct(h)} | {len(judged)} | "
          + ("'노트에 없는 관계 금지' 강화, 간선 검증 앞당김" if h > THRESHOLDS["hallucination"] else "유지") + " |")
    e = m["evidence_match"]
    print(f"| 근거 일치율 | {pct(e)} | {len(evi)} | "
          + ("근거 위치를 유사도로 찾는 방법 추가" if e < THRESHOLDS["evidence_match"] else "유지") + " |")
    w = m["wrong_merge"]
    print(f"| 잘못 합침 비율 | {pct(w)} | {len(merge)} | "
          + ("자동 병합 기준 올림" if w > THRESHOLDS["wrong_merge"] else "유지") + " |")
    hi, lo = THRESHOLDS["bridge_rate"]
    b = m["bridge_rate"]
    print(f"| **매개 개념 연결률** | {pct(b)} | {len(pair)} | "
          + ("지금 설계 유지" if b >= hi else "설계 재검토 회의" if b < lo else "보류 (20~50%)") + " |")

    print(f"\n놓친 관계: {missed}개")
    print("관계 라벨 분포:", {l: judged.count(l) for l in REL_LABELS if l != "놓친 관계"})
    print("매개 개념 분포:", {l: pair.count(l) for l in PAIR_LABELS})
    pending = {"relations(rel)": len(rel_pending), "relations(evi)": len(evi_pending),
               "merges": len(merge_pending), "pairs": len(pair_pending)}
    if any(pending.values()):
        print("\n미결 행 (채점 없음 또는 두 채점자 불일치, 지표에서 빠짐):", pending)

    # 틀린 사례 후보 (토요일 공유용 5~10개)
    wrong = [r for r in rel_rows if final_label(r, "rel") not in ("", "맞음")]
    if wrong:
        print("\n틀린 사례 후보:")
        for r in wrong[: args.examples]:
            label = final_label(r, "rel")
            print(f"  - [{label}] {r['note_id']}: {r['subject']} -{r['relation_type']}-> {r['object']} "
                  f"| 근거: {r['evidence'][:60]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="노트마다 추출을 N번 실행하고 기록한다")
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    e.add_argument("--repeats", type=int, default=2, help="같은 노트 반복 횟수 (흔들림 확인)")
    e.add_argument("--notes", nargs="*", help="일부 note_id 만 실행")
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("sheets", help="채점표 CSV 3개를 만든다")
    s.add_argument("--run", help="run_id (기본: 가장 최근)")
    s.add_argument("--repeat", type=int, default=1, help="채점할 반복 회차")
    s.add_argument("--force", action="store_true", help="이미 있는 채점표를 덮어쓴다")
    s.set_defaults(func=cmd_sheets)

    m = sub.add_parser("metrics", help="채점표로 지표와 판단을 출력한다")
    m.add_argument("--examples", type=int, default=10, help="틀린 사례 출력 개수")
    m.set_defaults(func=cmd_metrics)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
