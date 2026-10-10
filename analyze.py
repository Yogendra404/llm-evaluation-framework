"""Aggregate results, plot distributions, auto-diagnose worst cases, write REPORT.md."""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

METRICS = ["faithfulness", "answer_relevancy", "context_precision",
           "answer_correctness", "retrieval_hit"]
SCORED = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
N_WORST = int(os.getenv("N_WORST", "5"))
REFUSAL_MARKERS = ("don't have enough information", "do not have enough information",
                   "not enough information", "cannot find", "not mentioned", "not specified")

df = pd.read_csv(f"{config.RESULTS_DIR}/results.csv")
df["composite"] = df[SCORED].mean(axis=1, skipna=True)


def is_refusal(ans):
    a = str(ans).lower()
    return any(m in a for m in REFUSAL_MARKERS)


def fmt(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}"


def lowest_metric(r):
    vals = {m: r[m] for m in SCORED if not pd.isna(r[m])}
    return min(vals, key=vals.get) if vals else "n/a"


def diagnose(r, d):
    """Return (cause, explanation, proposed_fix) from the numbers. Verify by hand."""
    unans = r["reference"] == "NOT_IN_DOCS"
    refused = is_refusal(r["answer"])
    corr, hit = r["answer_correctness"], r["retrieval_hit"]
    unsupported = [v.get("claim", "") for v in
                   d.get("faithfulness", {}).get("verdicts", []) if not v.get("supported")]
    reason = d.get("correctness", {}).get("reason", "")

    if unans and not refused and (pd.isna(corr) or corr < 1):
        return ("Hallucination",
                "The documents contain no answer to this question, but the app produced one "
                "instead of declining. " + (f"Unsupported claims: {unsupported}. " if unsupported else ""),
                "Strengthen the refusal instruction, add a similarity threshold so weak "
                "retrieval returns no context, and add more unanswerable cases to the test set.")
    if unans and refused:
        return ("Metric artifact (correct refusal)",
                "The app correctly declined. The low composite comes from answer_relevancy "
                "= 0, because a refusal counts as non-committal.",
                "Exclude answer_relevancy for unanswerable cases (or report them separately). "
                "No change to the app is needed.")
    if not unans and not pd.isna(hit) and hit == 0:
        return ("Retrieval failure",
                f"The gold document was not among the retrieved sources ({r['sources']}), so "
                "the model never saw the needed facts. Judge note: " + reason,
                "Increase TOP_K, use a stronger embedding model, add hybrid BM25 + vector "
                "search, or rewrite the query before retrieval.")
    if not unans and refused:
        return ("Over-refusal",
                "The right document was retrieved, but the app still said it lacked the "
                "information, so the prompt or model was too cautious.",
                "Relax the refusal wording, ask the model to reason over the context for "
                "conditional questions, or use a stronger app model.")
    if not pd.isna(corr) and corr >= 0.99:
        low = lowest_metric(r)
        extra = f" Unsupported claims flagged by the judge: {unsupported}." if unsupported else ""
        return ("Metric / judge artifact",
                f"The answer matches the reference (correctness {fmt(corr)}). The low "
                f"composite is driven by {low}, not by a wrong answer." + extra,
                "Read the judge's verdicts by hand; tighten the judge prompt or use a larger "
                "judge model. Consider weighting correctness more heavily in the composite.")
    if not unans and not pd.isna(hit) and hit == 1:
        return ("Generation error",
                "The right chunk was retrieved, but the answer is wrong or incomplete. "
                "Judge note: " + reason,
                "Add a step-by-step reasoning instruction (important for multi-hop and "
                "arithmetic questions), give a worked example, or use a stronger app model.")
    return ("Unclear - inspect manually",
            "The pattern does not match a standard failure type. Judge note: " + reason,
            "Read the retrieved chunks and the answer side by side.")


# ---------- aggregates ----------
agg = df[METRICS].agg(["mean", "median", "std", "min", "max"]).T.round(3)
agg["% cases >= 0.8"] = [round(100 * (df[m].dropna() >= 0.8).mean(), 1) for m in METRICS]
by_cat = df.groupby("category")[METRICS + ["composite"]].mean().round(3)
by_cat = by_cat.join(df.groupby("category").size().rename("n"))

# ---------- plots ----------
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, m in zip(axes, SCORED):
    ax.hist(df[m].dropna(), bins=10, range=(0, 1), edgecolor="black")
    ax.set_title(m); ax.set_xlabel("score"); ax.set_ylabel("# cases")
plt.tight_layout()
plt.savefig(f"{config.RESULTS_DIR}/distributions.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4))
by_cat[SCORED].plot(kind="bar", ax=ax)
ax.set_ylim(0, 1.05); ax.set_title("Mean score by question category")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(f"{config.RESULTS_DIR}/by_category.png", dpi=150)

# ---------- diagnose worst cases ----------
worst = df.sort_values("composite").head(N_WORST)
diagnoses = []
for _, r in worst.iterrows():
    d = json.loads(r["details"])
    diagnoses.append((r, d) + diagnose(r, d))

cause_counts = pd.Series([x[2] for x in diagnoses]).value_counts()

# ---------- write REPORT.md ----------
L = []
L += ["# Evaluation Report\n",
      f"- **Test cases:** {len(df)}",
      f"- **App model:** `{config.APP_MODEL}`  |  **Judge model:** `{config.JUDGE_MODEL}`  |  **top-k:** {config.TOP_K}",
      f"- **Mean latency per question:** {df['latency_s'].mean():.2f}s\n"]

L += ["## 1. Methodology\n",
      "A fixed test set of real questions, each with a reference answer, is run through the RAG "
      "assistant. Every output is scored automatically by `run_eval.py` using these metrics:\n",
      "| Metric | How it is computed |", "|---|---|",
      "| Faithfulness | An LLM judge splits the answer into atomic claims and checks each against the retrieved context. Score = share of supported claims. |",
      "| Answer relevancy | An LLM writes 3 questions the answer would respond to; score = mean embedding cosine similarity to the real question. Refusals score 0. |",
      "| Context precision | An LLM marks each retrieved chunk useful or not; score = rank-aware average precision. Skipped for unanswerable cases. |",
      "| Answer correctness | An LLM compares the answer to the reference (0, 0.5 or 1). For unanswerable questions, declining is correct. |",
      "| Retrieval hit | Deterministic: was the gold source document among the retrieved chunks? |\n"]

L += ["## 2. Aggregate results\n", agg.to_markdown(), "\n",
      "### By question category\n", by_cat.to_markdown(), "\n",
      "![distributions](distributions.png)\n", "![by category](by_category.png)\n"]

L += [f"## 3. Failure analysis (the {len(diagnoses)} lowest-scoring cases)\n",
      "Diagnoses below are auto-drafted from the scores. **Verify each one by hand** before "
      "submitting: read the question, retrieved sources and answer, then confirm or edit the cause.\n",
      "### Summary of causes\n",
      "\n".join(f"- {c}: {n} case(s)" for c, n in cause_counts.items()), "\n"]

for r, d, cause, why, fix in diagnoses:
    L += [f"### Case {r['id']} ({r['category']}) - composite {fmt(r['composite'])}",
          f"- **Question:** {r['question']}",
          f"- **Reference answer:** {r['reference']}",
          f"- **App answer:** {r['answer']}",
          f"- **Retrieved sources:** {r['sources']}",
          f"- **Scores:** faithfulness={fmt(r['faithfulness'])}, relevancy={fmt(r['answer_relevancy'])}, "
          f"precision={fmt(r['context_precision'])}, correctness={fmt(r['answer_correctness'])}, "
          f"retrieval_hit={fmt(r['retrieval_hit'])}",
          f"- **Diagnosis:** {cause}",
          f"- **Analysis:** {why}",
          f"- **Proposed fix:** {fix}",
          "- [ ] Verified by hand (edit the diagnosis above if it is wrong)\n"]

L += ["## 4. Limitations\n",
      f"- The test set has only {len(df)} questions, so averages are noisy and small differences are not meaningful.",
      "- LLM judges can be wrong or inconsistent, especially smaller free-tier models; a sample of verdicts should be checked by hand.",
      "- If the same model acts as app and judge, the judge may favour its own style (self-preference bias).",
      "- Answer relevancy gives 0 to correct refusals, which understates performance on unanswerable questions.",
      "- Reference answers were written by hand and may themselves contain errors.\n",
      "## 5. Suggested improvements\n",
      "- Change one variable at a time (for example TOP_K, chunk size, prompt wording), rerun, and compare before/after scores.",
      "- Add a retrieval-similarity threshold so questions with no good match return a refusal.",
      "- Grow the test set with more hard, multi-hop and unanswerable cases.\n"]

open(f"{config.RESULTS_DIR}/REPORT.md", "w", encoding="utf-8").write("\n".join(L))
print(agg, "\n\n", by_cat, "\n\nDiagnosed worst cases:")
for r, d, cause, why, fix in diagnoses:
    print(f"  Case {r['id']:>2} [{r['category']}] composite={fmt(r['composite'])} -> {cause}")
print(f"\nSaved {config.RESULTS_DIR}/REPORT.md, distributions.png, by_category.png")