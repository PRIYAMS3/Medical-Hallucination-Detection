"""Evidence retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class EvidenceHit:
    """Retrieved evidence passage."""

    doc_id: str
    text: str
    score: float


class TfidfEvidenceRetriever:
    """Small BM25-like lexical baseline using TF-IDF cosine similarity."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=100_000)
        self._doc_ids: list[str] = []
        self._docs: list[str] = []
        self._matrix = None

    def fit(self, documents: list[str], doc_ids: list[str] | None = None) -> None:
        clean_docs = [doc for doc in documents if doc and doc.strip()]
        if doc_ids is None:
            clean_ids = [str(i) for i in range(len(clean_docs))]
        else:
            clean_ids = [doc_id for doc_id, doc in zip(doc_ids, documents) if doc and doc.strip()]

        self._docs = clean_docs
        self._doc_ids = clean_ids
        self._matrix = self._vectorizer.fit_transform(clean_docs)

    def search(self, query: str, top_k: int = 5) -> list[EvidenceHit]:
        if self._matrix is None:
            raise RuntimeError("Retriever is not fitted.")
        if not query.strip():
            return []

        query_vector = self._vectorizer.transform([_expand_medical_query(query)])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]

        return [
            EvidenceHit(
                doc_id=self._doc_ids[idx],
                text=self._docs[idx],
                score=float(scores[idx]),
            )
            for idx in ranked
        ]


def _expand_medical_query(query: str) -> str:
    """Add a few high-value medical synonyms for the lexical baseline."""

    expansions = {
        "kidney failure": "renal failure renal impairment kidney disease",
        "renal failure": "kidney failure renal impairment kidney disease",
        "pregnant": "pregnancy gestation maternal",
        "cure": "treat treatment management therapy",
        "cures": "treats treatment management therapy",
        "diabetes": "diabetic glycemic glucose insulin",
    }

    expanded = query
    lowered = query.lower()
    for term, synonyms in expansions.items():
        if term in lowered:
            expanded = f"{expanded} {synonyms}"
    return expanded
