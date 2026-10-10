"""Run the full test set through the app and score every output."""
import argparse, json, os, time
import pandas as pd
from tqdm import tqdm

import config, judge
from rag_app import RAGApp


def safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # keep the run alive; record the error
        return float("nan"), {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run first N cases")
    args = ap.parse_args()

    tests = json.load(open(config.TESTSET, encoding="utf-8"))
    if args.limit:
        tests = tests[: args.limit]

    app = RAGApp()
    rows = []
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for t in tqdm(tests, desc="Evaluating"):
        t0 = time.time()
        out = app.query(t["question"])
        latency = time.time() - t0
        unanswerable = t["reference"] == "NOT_IN_DOCS"

        faith, fd = safe(judge.faithfulness, t["question"], out["answer"], out["contexts"])
        rel, rd = safe(judge.answer_relevancy, t["question"], out["answer"], app.embedder)
        corr, cd = safe(judge.answer_correctness, t["question"], out["answer"], t["reference"])
        if unanswerable:
            prec, pd_, hit = float("nan"), {}, float("nan")
        else:
            prec, pd_ = safe(judge.context_precision, t["question"], t["reference"], out["contexts"])
            hit = int(t["source"] in out["sources"])

        rows.append({
            "id": t["id"], "category": t["category"], "question": t["question"],
            "reference": t["reference"], "answer": out["answer"],
            "sources": ", ".join(out["sources"]), "latency_s": round(latency, 2),
            "faithfulness": faith, "answer_relevancy": rel,
            "context_precision": prec, "answer_correctness": corr,
            "retrieval_hit": hit,
            "details": json.dumps({"faithfulness": fd, "relevancy": rd,
                                   "precision": pd_, "correctness": cd},
                                  ensure_ascii=False),
        })
        pd.DataFrame(rows).to_csv(f"{config.RESULTS_DIR}/results.csv", index=False)  # checkpoint

    print(f"\nDone. Saved {len(rows)} rows to {config.RESULTS_DIR}/results.csv")


if __name__ == "__main__":
    main()