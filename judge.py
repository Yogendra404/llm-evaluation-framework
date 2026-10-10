"""LLM-as-judge metrics (RAGAS-style, implemented transparently).

faithfulness       : fraction of claims in the answer supported by retrieved context
answer_relevancy   : similarity between the question and questions regenerated from the answer
context_precision  : rank-aware precision of retrieved chunks (Average Precision)
answer_correctness : judge compares answer to the reference answer (0 / 0.5 / 1)
retrieval_hit      : deterministic - was the gold source document retrieved? (in run_eval.py)
"""
import json
import numpy as np

import config
from llm import call_llm, parse_json

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluation judge. Respond with valid JSON only - "
    "no prose, no markdown fences."
)


def _judge(prompt):
    return parse_json(call_llm(prompt, config.JUDGE_MODEL, system=JUDGE_SYSTEM))


def _ctx(contexts):
    return "\n\n".join(f"[Chunk {i + 1}] {c}" for i, c in enumerate(contexts))


# 1. Faithfulness -----------------------------------------------------------
def faithfulness(question, answer, contexts):
    claims = _judge(
        "Break the ANSWER into atomic, self-contained factual claims. If the answer "
        "only says it lacks information / declines, return an empty list.\n"
        'Return {"claims": ["..."]}.\n\n'
        f"QUESTION: {question}\nANSWER: {answer}"
    ).get("claims", [])
    if not claims:
        return 1.0, {"claims": [], "verdicts": []}

    verdicts = _judge(
        "For each CLAIM decide whether it can be directly inferred from the CONTEXT "
        "alone. Ignore world knowledge. A claim that is true in reality but absent "
        "from the context is NOT supported.\n"
        'Return {"verdicts": [{"claim": "...", "supported": true, "reason": "..."}]}.\n\n'
        f"CONTEXT:\n{_ctx(contexts)}\n\nCLAIMS:\n{json.dumps(claims, ensure_ascii=False)}"
    ).get("verdicts", [])
    if not verdicts:
        return float("nan"), {"claims": claims, "verdicts": []}
    score = float(np.mean([bool(v["supported"]) for v in verdicts]))
    return score, {"claims": claims, "verdicts": verdicts}


# 2. Answer relevancy -------------------------------------------------------
def answer_relevancy(question, answer, embedder, n=3):
    out = _judge(
        f"Given the ANSWER, write {n} different questions that this answer would be a "
        'direct response to. Also set "noncommittal" to true if the answer is evasive '
        "or says it does not know.\n"
        'Return {"questions": ["..."], "noncommittal": false}.\n\n'
        f"ANSWER: {answer}"
    )
    if out.get("noncommittal"):
        return 0.0, out
    vecs = embedder.encode([question] + out["questions"], normalize_embeddings=True)
    sims = vecs[1:] @ vecs[0]
    return float(max(0.0, np.mean(sims))), out


# 3. Context precision ------------------------------------------------------
def context_precision(question, reference, contexts):
    out = _judge(
        "For each retrieved chunk, decide if it contains information useful for "
        "producing the REFERENCE ANSWER to the QUESTION.\n"
        'Return {"relevant": [true, false, ...]} with exactly one boolean per chunk, '
        "in order.\n\n"
        f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\n\n{_ctx(contexts)}"
    )
    rels = [1 if r else 0 for r in out["relevant"]][: len(contexts)]
    if sum(rels) == 0:
        return 0.0, out
    ap = sum((sum(rels[: i + 1]) / (i + 1)) * r for i, r in enumerate(rels)) / sum(rels)
    return float(ap), out


# 4. Answer correctness -----------------------------------------------------
def answer_correctness(question, answer, reference):
    out = _judge(
        "Compare the ANSWER with the REFERENCE ANSWER for the QUESTION.\n"
        "Scoring: 1.0 = all key facts correct; 0.5 = partially correct or misses a key "
        "detail; 0.0 = wrong, contradictory, or fabricated.\n"
        'If REFERENCE is "NOT_IN_DOCS", the correct behaviour is to decline to answer; '
        "any invented answer scores 0.0.\n"
        'Return {"score": 1.0, "reason": "..."}.\n\n'
        f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}"
    )
    return float(out["score"]), out