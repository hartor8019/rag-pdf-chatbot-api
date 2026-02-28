import json
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/ask"

def score_answer(answer: str, must_include: list[str]) -> dict:
    a = answer.lower()
    hits = [kw for kw in must_include if kw.lower() in a]
    misses = [kw for kw in must_include if kw.lower() not in a]
    return {
        "pass": len(misses) == 0,
        "hits": hits,
        "misses": misses,
        "coverage": round(len(hits) / max(1, len(must_include)), 2),
    }

def main():
    golden_path = Path("eval/golden_set.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    results = []
    for item in golden:
        payload = {"question": item["question"], "top_k": 4}
        r = requests.post(API_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "")

        s = score_answer(answer, item["must_include"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "pass": s["pass"],
            "coverage": s["coverage"],
            "misses": s["misses"],
        })
        print(f"{item['id']} | pass={s['pass']} | coverage={s['coverage']} | misses={s['misses']}")

    passed = sum(1 for x in results if x["pass"])
    print(f"\nTOTAL: {passed}/{len(results)} passed")

    Path("eval/report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved eval/report.json")

if __name__ == "__main__":
    main()