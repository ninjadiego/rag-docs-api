"""Retrieval-Augmented Generation pipeline: retrieve → build prompt → answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm import ChatModel, Completion
from app.store import Hit, Retriever

SYSTEM_PROMPT = """You answer questions using only the provided context.
Rules:
- If the answer is not in the context, say you don't know. Do not guess.
- Cite sources inline as [n] using the numbers of the context passages you used.
- Be concise and answer in the language of the question."""

NO_CONTEXT_ANSWER = "I don't have any documents that cover this question yet."


@dataclass(frozen=True)
class Source:
    n: int
    document_id: str
    chunk_index: int
    score: float
    excerpt: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]
    model: str | None
    prompt_tokens: int
    completion_tokens: int


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = ["Context:"]
    for n, h in enumerate(hits, start=1):
        parts.append(f"[{n}] (doc={h.document_id}, chunk={h.chunk_index})\n{h.text}")
    parts.append(f"\nQuestion: {question}")
    return "\n\n".join(parts)


def _score(distance: float) -> float:
    """Cosine distance in [0, 2] → similarity in [0, 1], rounded for display."""
    return round(max(0.0, 1.0 - distance), 4)


class RAGPipeline:
    def __init__(self, retriever: Retriever, model: ChatModel, top_k: int = 4) -> None:
        self.retriever = retriever
        self.model = model
        self.top_k = top_k

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.retriever.query(question, top_k or self.top_k)
        sources = [
            Source(
                n=n,
                document_id=h.document_id,
                chunk_index=h.chunk_index,
                score=_score(h.distance),
                excerpt=h.text[:200],
                metadata=h.metadata,
            )
            for n, h in enumerate(hits, start=1)
        ]
        if not hits:
            return Answer(NO_CONTEXT_ANSWER, [], None, 0, 0)

        completion: Completion = self.model.complete(SYSTEM_PROMPT, build_prompt(question, hits))
        return Answer(
            answer=completion.text.strip(),
            sources=sources,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )
