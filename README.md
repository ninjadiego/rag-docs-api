# rag-docs-api

> Ask questions about your own documents and get answers that cite the passages they came from.
> FastAPI + ChromaDB, with the LLM served through [go-ai-gateway](https://github.com/ninjadiego/go-ai-gateway) so every call has a budget and a cost.

[![CI](https://github.com/ninjadiego/rag-docs-api/actions/workflows/ci.yml/badge.svg)](https://github.com/ninjadiego/rag-docs-api/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.x-FF6F00)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What it does

Upload text or Markdown, then ask questions. The service splits documents into overlapping chunks, embeds them into a local vector store, retrieves the most relevant passages for each question and asks the model to answer **only from those passages**, citing them as `[1]`, `[2]`… If nothing relevant is stored, it says so instead of inventing an answer.

```
POST /documents            ──▶  chunk ──▶ embed ──▶ ChromaDB
POST /query  ──▶ embed question ──▶ top-k passages ──▶ prompt ──▶ LLM (via gateway) ──▶ answer + sources
```

The LLM client speaks the OpenAI chat-completions format, so `LLM_BASE_URL` can be the gateway, OpenAI itself, or a local server such as Ollama or vLLM. With the gateway in front you get per-key rate limits, a monthly budget and the USD cost of every answer for free.

---

## Quickstart

```bash
git clone https://github.com/ninjadiego/rag-docs-api.git
cd rag-docs-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # set LLM_BASE_URL / LLM_API_KEY (a gw_live_... key from go-ai-gateway)
make dev                      # http://localhost:8000/docs
```

```bash
# Ingest
curl -s -X POST localhost:8000/documents -H 'Content-Type: application/json' \
  -d '{"name":"policy.md","text":"Refunds are accepted within 30 days of purchase with a receipt.","metadata":{"team":"cx"}}'
# → {"document_id":"9f1c…","name":"policy.md","chunks":1}

# Or upload a file
curl -s -X POST localhost:8000/documents/upload -F file=@handbook.md

# Ask
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"How many days do I have to ask for a refund?"}'
```

```json
{
  "answer": "You have 30 days from the purchase date, and you need the receipt [1].",
  "sources": [
    {"n": 1, "document_id": "9f1c…", "chunk_index": 0, "score": 0.83,
     "excerpt": "Refunds are accepted within 30 days of purchase with a receipt.",
     "metadata": {"name": "policy.md", "team": "cx"}}
  ],
  "model": "claude-sonnet-4-6",
  "usage": {"prompt_tokens": 214, "completion_tokens": 27, "total_tokens": 241}
}
```

Docker:

```bash
docker compose up --build -d    # reaches a gateway on the host at :8080 by default
```

---

## API

| Method | Path                       | Purpose                                                       |
|--------|----------------------------|---------------------------------------------------------------|
| GET    | `/health`                  | Liveness + number of stored documents                         |
| POST   | `/documents`               | Ingest a document from JSON (`name`, `text`, optional `metadata`) |
| POST   | `/documents/upload`        | Ingest a `.txt` / `.md` file (multipart, ≤ 2 MB, UTF-8)       |
| DELETE | `/documents/{document_id}` | Remove a document and all its chunks                          |
| POST   | `/query`                   | Answer a question with citations (`question`, optional `top_k`) |

Interactive docs at `/docs` (Swagger) and `/redoc`.

Document ids are derived from the name, so re-uploading `policy.md` **replaces** its chunks instead of duplicating them.

---

## Project structure

```
rag-docs-api/
├── app/
│   ├── main.py        # FastAPI routes, request/response schemas, dependency wiring
│   ├── config.py      # pydantic-settings (env / .env)
│   ├── chunking.py    # overlapping windows that snap to paragraph / sentence / word boundaries
│   ├── store.py       # ChromaDB wrapper: upsert by document id, cosine query, delete
│   ├── llm.py         # OpenAI-compatible chat client (works with go-ai-gateway)
│   └── rag.py         # retrieve → prompt → answer, with numbered citations
├── tests/             # 21 tests, no network, no model download
├── .github/workflows/ci.yml   # ruff + pytest on Python 3.11 and 3.12
├── Dockerfile · docker-compose.yml · Makefile
```

---

## Development

```bash
make test     # pytest — runs in ~1 s
make lint     # ruff check + format check
make fmt      # auto-format
```

Tests never touch the network: they use an ephemeral Chroma collection, a deterministic hashing embedding function (so CI does not download an embedding model) and a fake chat model that records the prompt it receives. The pipeline is built from two small protocols (`Retriever`, `ChatModel`), which is what makes that possible.

---

## Design decisions

- **Grounded answers or nothing.** The system prompt forbids answering outside the context, and the pipeline short-circuits with a fixed message when retrieval returns nothing, so the model is never asked to improvise.
- **Citations are structural, not cosmetic.** Passages are numbered in the prompt and returned as `sources` with the same numbers, score and excerpt, so a UI can link each `[n]` to the text behind it.
- **Chunk boundaries matter more than chunk size.** Windows snap back to a paragraph break if one exists in the last half of the window, then to a sentence end, then to a word; overlaps also start on a word. Cutting mid-sentence is the most common cause of bad retrieval in small RAG systems.
- **Upsert by document id.** Ids are `<document_id>:<chunk_index>`; re-ingesting first deletes the document's chunks, so the store never drifts after edits.
- **The LLM is a config value.** Only the OpenAI wire format is assumed. Through go-ai-gateway that means Claude with a budget; pointing at a local model is a one-line change.
- **Default embeddings are local.** Chroma's built-in `all-MiniLM-L6-v2` (ONNX) runs on CPU with no API key. Swap `embedding_function` for a hosted one when quality matters more than cost.

---

## Roadmap

- [ ] PDF and DOCX ingestion (text extraction)
- [ ] Streaming answers (SSE) once the gateway supports it on `/v1/chat/completions`
- [ ] Hybrid retrieval (BM25 + vectors) with reciprocal-rank fusion
- [ ] Reranking step for `top_k > 4`
- [ ] Conversation memory (follow-up questions)
- [ ] Minimal web UI

---

## License

MIT © 2026 Diego Peña

Second project in a two-part portfolio: [go-ai-gateway](https://github.com/ninjadiego/go-ai-gateway) (Go) handles keys, limits and cost; this service (Python) builds a product on top of it.
