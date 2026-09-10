"""Shared fixtures.

Everything runs in-process with no network: an ephemeral Chroma collection,
a deterministic embedding function (no model download) and a fake LLM.
"""

from __future__ import annotations

import hashlib
import math
import uuid

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from fastapi.testclient import TestClient

from app.llm import Completion
from app.main import app
from app.rag import RAGPipeline
from app.store import VectorStore


class HashingEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words hashed into a fixed-size vector.

    Not semantically meaningful, but deterministic and good enough for
    lexical overlap, which is what the tests assert on.
    """

    DIM = 256

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (chroma's signature)
        out = []
        for text in input:
            vec = [0.0] * self.DIM
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM  # noqa: S324
                vec[h] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def name() -> str:
        return "hashing-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> HashingEmbedding:
        return HashingEmbedding()


class FakeModel:
    """Records the prompt it received and returns a canned answer."""

    def __init__(self, reply: str = "The answer is 42 [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(text=self.reply, model="fake-model", prompt_tokens=100, completion_tokens=10)


@pytest.fixture
def store() -> VectorStore:
    # Chroma's ephemeral client is a process-wide singleton, so each test gets
    # its own collection name to stay isolated.
    return VectorStore.in_memory(
        collection_name=f"test-{uuid.uuid4().hex[:8]}",
        embedding_function=HashingEmbedding(),
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def pipeline(store: VectorStore, fake_model: FakeModel) -> RAGPipeline:
    return RAGPipeline(store, fake_model, top_k=3)


@pytest.fixture
def client(pipeline: RAGPipeline) -> TestClient:
    app.state.pipeline = pipeline
    with TestClient(app) as c:
        yield c
    del app.state.pipeline
