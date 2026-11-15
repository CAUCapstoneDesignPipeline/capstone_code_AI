"""풀링 판정: 정답에 없는 예측 관계를 노트에 비춰 한 번씩 판정해 모아 두고(캐시), 여러 방법이 같은 '맞는 관계' 집합으로 채점되게 한다.

정답(gold.py)은 노트마다 20~27개뿐이라, 방법이 노트가 실제로 말하는 관계를 더 찾으면 정밀도가 깎인다 (Sonnet 의 '오답' 표본의 70% 가량이 노트가 말하는 관계였다).
그래서 여러 방법의 예측을 합친 풀에서 정답에 없는 것을 판정자(Opus 5.5)가 노트를 보고 엄격하게 판정한다: 노트가 그 관계를 말하는가, 어느 유형이 맞는가, 방향은.
맞는 관계 집합 V = 정답 ∪ (판정자가 맞다고 본 것, 판정자가 고른 유형·방향). 방법의 정밀도 = 낸 관계 중 V 에 든 비율, 재현율 = V 중 낸 비율.
판정자가 정답 초안을 쓴 모델과 같은 계열이라 정답과 판단이 닮는다 (독립 검증이 아니다). 판정 저장: data/private/gold-relations/adjudications.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data/private/gold-relations/adjudications.json"
CHUNK = 60


def canon(rel):
    """(source, target, type) → 비교용 키. 대칭 유형은 두 끝을 정렬한다."""
    from experiments.relations.harness import SYMMETRIC
    s, t, ty = rel
    return (tuple(sorted((s, t))) if ty in SYMMETRIC else (s, t), ty)


class Pool:
    def __init__(self):
        self.store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}      # "set/note" → {"s|t|type": {valid, best_type, swap}}
        self._judge = None

    def judge(self):
        if self._judge is None:
            from dataclasses import replace

            from app.main import role_llm_config
            from app.services.llm import make_llm
            from experiments.relations import gold as G
            self._judge = make_llm(replace(role_llm_config("relation", "medium"), provider="anthropic", model=G.DRAFT_MODEL, effort="medium"))
        return self._judge

    def save(self):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(self.store, ensure_ascii=False), encoding="utf-8")

    def need(self, name, note, preds):
        """아직 판정하지 않은, 정답에 없는 예측."""
        gold = {canon((r["source"], r["target"], r["type"])) for r in note.gold}
        have = self.store.get(f"{name}/{note.name}", {})
        return [r for r in dict.fromkeys(preds) if canon(r) not in gold and "|".join(r) not in have]

    def adjudicate(self, jobs, workers=4):
        """jobs: [(집합, 노트, 예측들)] 의 판정되지 않은 것을 Opus 로 판정해 저장한다."""
        from concurrent.futures import ThreadPoolExecutor

        from experiments.relations import adjudicate as A
        todo = []
        for name, note, preds in jobs:
            need = self.need(name, note, preds)
            for i in range(0, len(need), CHUNK):
                todo.append((name, note, need[i:i + CHUNK]))
        if not todo:
            return 0

        def run(item):
            name, note, chunk = item
            listing = "\n".join(f"[{i}] {r[0]} --{r[2]}--> {r[1]}" for i, r in enumerate(chunk))
            user = "<개념 목록>\n" + "\n".join(f"- {c}" for c in note.concepts) + f"\n</개념 목록>\n\n<노트>\n{note.text}\n</노트>\n\n<판정할 관계>\n{listing}\n</판정할 관계>"
            for attempt in range(4):
                try:
                    res = self.judge().complete_json(A.SYSTEM, user, A.SCHEMA, "adjudicate").data
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    import time
                    time.sleep(5 * (attempt + 1))
            return name, note, chunk, {j["id"]: j for j in res["judgments"]}

        n = 0
        with ThreadPoolExecutor(workers) as pool:
            for name, note, chunk, js in pool.map(run, todo):
                rec = self.store.setdefault(f"{name}/{note.name}", {})
                for i, r in enumerate(chunk):
                    j = js.get(i, {"valid": False, "best_type": r[2], "swap": False})
                    rec["|".join(r)] = {"valid": bool(j["valid"]), "best_type": j["best_type"], "swap": bool(j["swap"])}
                    n += 1
        self.save()
        return n

    def valid_set(self, name, note):
        """V: 정답 ∪ 판정자가 맞다고 본 관계 (판정자가 고른 유형·방향으로 정리)."""
        from experiments.relations.harness import SYMMETRIC
        v = {canon((r["source"], r["target"], r["type"])) for r in note.gold}
        for k, j in self.store.get(f"{name}/{note.name}", {}).items():
            if j["valid"]:
                s, t, _ = k.split("|")
                best = j["best_type"]
                v.add(canon((t, s, best) if (j["swap"] and best not in SYMMETRIC) else (s, t, best)))
        return v

    def valid_pairs(self, name, note):
        from experiments.relations.harness import SYMMETRIC  # noqa: F401
        pairs = {frozenset((r["source"], r["target"])) for r in note.gold}
        for k, j in self.store.get(f"{name}/{note.name}", {}).items():
            if j["valid"]:
                s, t, _ = k.split("|")
                pairs.add(frozenset((s, t)))
        return pairs
