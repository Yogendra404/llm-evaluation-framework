"""The application under test: a small RAG assistant over company policy docs.
Swap this class for YOUR app (RAG system / chatbot) - the only contract is
`query(question) -> {"answer": str, "contexts": [str], "sources": [str]}`.
"""
import glob, os
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from llm import call_llm

SYSTEM = (
    "You are an HR/policy assistant for Zentra Labs employees. Answer ONLY using "
    "the provided context. If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information in the provided documents to answer that.\" "
    "Be concise and include specific numbers when they are in the context."
)


class RAGApp:
    def __init__(self, docs_dir=config.DOCS_DIR, k=config.TOP_K):
        self.k = k
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.chunks = self._load(docs_dir)
        self.vecs = self.embedder.encode(
            [c["text"] for c in self.chunks], normalize_embeddings=True
        )

    @staticmethod
    def _load(docs_dir):
        chunks = []
        for path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
            name = os.path.basename(path)
            text = open(path, encoding="utf-8").read()
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            buf = ""
            for p in paras:
                if len(buf) + len(p) < 600:
                    buf = (buf + "\n" + p).strip()
                else:
                    if buf:
                        chunks.append({"source": name, "text": f"[{name}] {buf}"})
                    buf = p
            if buf:
                chunks.append({"source": name, "text": f"[{name}] {buf}"})
        return chunks

    def retrieve(self, question):
        qv = self.embedder.encode([question], normalize_embeddings=True)[0]
        scores = self.vecs @ qv
        idx = np.argsort(-scores)[: self.k]
        return [self.chunks[i] for i in idx]

    def query(self, question):
        hits = self.retrieve(question)
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        answer = call_llm(prompt, config.APP_MODEL, system=SYSTEM, max_tokens=1500)
        return {
            "answer": answer.strip(),
            "contexts": [h["text"] for h in hits],
            "sources": [h["source"] for h in hits],
        }