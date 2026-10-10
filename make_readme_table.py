import pandas as pd
import config

df = pd.read_csv(f"{config.RESULTS_DIR}/results.csv")

rows = [("Faithfulness", "faithfulness"),
        ("Answer relevancy", "answer_relevancy"),
        ("Context precision", "context_precision"),
        ("Answer correctness", "answer_correctness"),
        ("Retrieval hit rate", "retrieval_hit")]

scored = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
df["composite"] = df[scored].mean(axis=1, skipna=True)
by_cat = df.groupby("category")["composite"].mean().sort_values()
weakest, strongest = by_cat.index[0], by_cat.index[-1]

print(f"Evaluated on {len(df)} test questions ({config.APP_MODEL} as the assistant, "
      f"{config.JUDGE_MODEL} as the judge, top-k = {config.TOP_K}):\n")
print("| Metric | Mean |")
print("|---|---|")
for label, col in rows:
    print(f"| {label} | {df[col].mean():.2f} |")
print(f"\nThe weakest question category was **{weakest}** "
      f"(mean composite {by_cat[weakest]:.2f}) and the strongest was **{strongest}** "
      f"(mean composite {by_cat[strongest]:.2f}).")