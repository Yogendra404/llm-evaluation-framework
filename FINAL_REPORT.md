# Evaluating a RAG Assistant with LLM-as-Judge Metrics

## 1. Application under test

A retrieval-augmented question-answering assistant over 8 company policy documents (leave, remote work, expenses, onboarding, learning, security, benefits, performance).

| Component | Choice |
|---|---|
| Documents | 8 markdown files, chunked by paragraph (about 600 characters per chunk) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | Cosine similarity, top-k = 3 |
| Generator (app model) | Groq `openai/gpt-oss-120b` |
| Judge model | Groq `openai/gpt-oss-120b` |
| Prompt rule | Answer only from context; otherwise reply that the information is not available |

The same model (`openai/gpt-oss-120b`) acted as both the assistant and the judge, because a free-tier setup was used. This may introduce self-preference bias in the judge's scores.

## 2. Test set

21 questions with hand-written reference answers and a gold source document.

| Category | Count | Purpose |
|---|---|---|
| easy | 10 | Single-fact lookups |
| conditional | 6 | Rules with thresholds or conditions |
| multi_hop | 2 | Combining facts or doing arithmetic |
| unanswerable | 3 | Information is not in the documents; the correct behaviour is to decline |

## 3. Metrics

All metrics are computed automatically by `run_eval.py`; nothing was scored by eye.

| Metric | What it measures | How it is computed |
|---|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? | A judge splits the answer into atomic claims and checks each against the context. Score = supported claims / total claims. |
| Answer relevancy | Does the answer address the question? | A judge writes 3 questions the answer would respond to; score = mean cosine similarity to the real question. Refusals score 0. |
| Context precision | Are the relevant chunks ranked high? | A judge marks each chunk useful or not; score = average precision. Skipped for unanswerable cases. |
| Answer correctness | Does the answer match the reference? | A judge scores 0, 0.5 or 1 against the reference. For unanswerable questions, declining scores 1. |
| Retrieval hit | Was the right document retrieved? | Deterministic check of the gold source against the retrieved sources. |

## 4. Aggregate results

Full run on all 21 test questions.

| Metric | Mean | Median | Std | Min | Max | % of cases >= 0.8 |
|---|---|---|---|---|---|---|
| Faithfulness | 0.87 | 1.00 | 0.32 | 0.00 | 1.00 | 85.7 |
| Answer relevancy | 0.64 | 0.77 | 0.35 | 0.00 | 0.95 | 47.6 |
| Context precision | 0.86 | 1.00 | 0.29 | 0.00 | 1.00 | 77.8 |
| Answer correctness | 0.88 | 1.00 | 0.27 | 0.00 | 1.00 | 81.0 |
| Retrieval hit rate | 0.94 | 1.00 | 0.24 | 0.00 | 1.00 | 94.4 |

### By category

Mean score per category. Composite is the mean of the four judged metrics that apply to that question. Context precision and retrieval hit do not apply to unanswerable questions.

| Category | n | Faithfulness | Answer relevancy | Context precision | Retrieval hit | Composite |
|---|---|---|---|---|---|---|
| conditional | 6 | 0.83 | 0.79 | 1.00 | 1.00 | 0.91 |
| easy | 10 | 1.00 | 0.72 | 0.75 | 0.90 | 0.81 |
| multi_hop | 2 | 0.17 | 0.75 | 1.00 | 1.00 | 0.73 |
| unanswerable | 3 | 1.00 | 0.00 | n/a | n/a | 0.67 |

### Distributions

![Score distributions](results/distributions.png)

![Scores by category](results/by_category.png)

### Observations

- Answer correctness (0.88), faithfulness (0.87) and context precision (0.86) are close together and high, so the assistant is mostly right, mostly grounded, and mostly retrieves useful chunks.
- The scores are skewed, not uniformly high. Faithfulness, context precision and correctness all have a median of 1.00 but a minimum of 0.00 and a standard deviation of 0.27 to 0.32, so most questions are answered perfectly while a few fail badly.
- Answer relevancy is the weakest metric at 0.64, and only 47.6% of cases reach 0.8. This is largely a property of the metric: it scores a correct refusal as 0, and 3 of the 21 questions are unanswerable.
- Retrieval hit rate is 0.94, which on the 18 answerable questions means 17 retrieved the right document and 1 missed it (Case 10).
- The unanswerable category has the lowest composite (0.67) even though all 3 answers were correct refusals, which I verified by reading them. The low composite comes from answer relevancy scoring refusals as 0.
- Multi-hop faithfulness is 0.17 on only 2 questions, while both answers were correct. This points to a limitation of the faithfulness metric on reasoning questions (see Section 5), not to wrong answers.
- Four questions had correctness below 1: Case 10 (0.0, a retrieval failure) and Cases 1, 5 and 6 (0.5 each). In Cases 1, 5 and 6 the answers were correct on the fact asked, and the judge deducted points only for extra detail in my reference answers that the question did not require (see Section 5). Had those three scored 1.0, mean correctness would have been about 0.95 instead of 0.88.
## 5. Failure case analysis

### Case 10 (easy): retrieval failure

- **Question:** How much internet reimbursement do I get monthly?
- **Reference:** ₹1,000 per month, on submission of the bill.
- **App answer:** "I don't have enough information in the provided documents to answer that."
- **Retrieved sources:** learning.md, benefits.md, expenses.md
- **Scores:** correctness 0.0, retrieval hit 0, composite 0.25
- **Cause:** Retrieval failure. The correct document (remote_work.md) ranked 5th by similarity (0.233), below the top-3 cutoff (0.298), so the model never saw the answer and correctly declined.
- **Why it happened:** The top two chunks (learning.md at 0.417, benefits.md at 0.413) discuss other reimbursements, which likely matched the word "reimbursement" in the question. The internet fact is one sentence inside a larger remote-work chunk, so its embedding may be diluted.
- **Proposed fix (not tested):** Smaller chunks with one fact each, a higher top-k (5 would have included the right document), hybrid keyword and vector search, or query rewriting.

### Case 17 (multi-hop): metric artifact

- **Question:** I'm an L4 engineer with a base salary of ₹20,00,000 and a rating of 4. What is my target bonus?
- **Reference:** 15% of base for L4, i.e. ₹3,00,000; eligible because rating is 3 or higher.
- **App answer:** "Your target annual bonus is 15% of ₹20,00,000, which equals ₹3,00,000."
- **Retrieved sources:** performance.md, benefits.md, expenses.md
- **Scores:** faithfulness 0.0, correctness 1.0, composite 0.66
- **Cause:** Metric artifact. The answer is correct, but the judge marked both of its claims as unsupported.
- **Why it happened:** The salary comes from the question and the ₹3,00,000 figure is computed, so neither appears in the retrieved context. The faithfulness check only compares claims with retrieved text, so it penalises correct reasoning.
- **Proposed fix (not tested):** Give the judge the question as well as the context, allow simple arithmetic from context facts, or weight correctness more heavily for multi-hop questions.

### Case 18 (multi-hop): metric artifact

- **Question:** Can I work remotely from my hometown for 6 weeks this year?
- **Reference:** No. Remote work outside the city is capped at 30 days per calendar year with manager approval; 6 weeks exceeds that.
- **App answer:** The app concluded that this is not allowed, reasoning that six weeks is about 42 days, more than the 30-day limit.
- **Scores:** faithfulness 0.33, correctness 1.0
- **Cause:** Metric artifact, the same pattern as Case 17. The judge flagged "six weeks is approximately 42 days" and the conclusion built on it as unsupported. Both are reasoning steps, not text copied from the documents.
- **Proposed fix (not tested):** Same as Case 17.


### Cases 1, 5 and 6 (easy): over-specified references

- **Questions:** annual leave days (Case 1), annual learning budget (Case 5), how quickly a lost laptop must be reported (Case 6).
- **App answers:** "24 days of annual leave per year"; "₹40,000 per employee annually"; "within 2 hours".
- **Scores:** correctness 0.5 each, with the correct document retrieved in all three.
- **Cause:** Metric and reference design artifact. The answers were correct on the fact asked. The judge deducted points for details present in my reference answers but not required by the question: the accrual rate (Case 1), the manager-approval rule (Case 5), and the email address (Case 6).
- **Why it happened:** I wrote the references as full policy summaries, and the judge prompt treats missing reference detail as a partial answer. In Case 6 the judge's own reason says the omitted email "is not required by the question", yet it still gave 0.5, so the judge applied its rule inconsistently.
- **Proposed fix (not tested):** Shorten references to the single key fact the question asks for, or tell the judge to score 1.0 when the key fact is correct and extra reference detail is optional.
### Pattern across failures

Only one failure was a real pipeline problem (Case 10, retrieval). The other low scores came from the evaluation design: faithfulness penalised correct reasoning in both multi-hop questions (mean 0.17, with correct answers), correctness penalised correct answers that left out optional reference detail (Cases 1, 5, 6), and answer relevancy scores correct refusals as 0. So retrieval quality limits answer quality, while several of the lowest scores say more about the metrics and references than about the assistant.

## 6. Judge validation

- **Cases 1, 5 and 6:** I read the judge's reasons. The answers were correct on the fact asked, and the judge deducted for optional detail from my references. In Case 6 the judge said the omitted detail was not required but still gave 0.5, so I disagreed with its scoring on all three.

This shows the judge was reliable on refusals and retrieval, but unreliable on faithfulness for multi-hop reasoning questions and too strict on correctness when references contain optional detail.
## 7. Limitations

- Only 21 test questions, so averages are noisy and small differences are not meaningful. The multi-hop category has only 2 questions.
- LLM judges can be inconsistent, especially when the judge is the same model as the assistant.
- Using one model as both assistant and judge may cause self-preference bias.
- Answer relevancy scores correct refusals as 0, which understates performance on unanswerable questions.
- Faithfulness penalises correct arithmetic and reasoning that is not stated verbatim in the context.
- Reference answers were written by hand and may contain mistakes.
- The proposed fixes were not tested.
- Some reference answers contain more detail than the question requires, which made the correctness judge deduct points for correct answers.

## 8. How to reproduce

```powershell
pip install openai sentence-transformers numpy pandas matplotlib tqdm tabulate
$env:PROVIDER="groq"
$env:GROQ_API_KEY="<your key>"
python run_eval.py
python analyze.py
```