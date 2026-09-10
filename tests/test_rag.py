from app.rag import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, RAGPipeline, build_prompt
from app.store import Hit


def test_build_prompt_numbers_passages():
    hits = [Hit("alpha text", "d1", 0, 0.1, {}), Hit("beta text", "d2", 3, 0.2, {})]
    prompt = build_prompt("what?", hits)
    assert "[1] (doc=d1, chunk=0)\nalpha text" in prompt
    assert "[2] (doc=d2, chunk=3)\nbeta text" in prompt
    assert prompt.endswith("Question: what?")


def test_ask_without_documents_does_not_call_model(pipeline: RAGPipeline, fake_model):
    ans = pipeline.ask("anything?")
    assert ans.answer == NO_CONTEXT_ANSWER
    assert ans.sources == [] and ans.model is None
    assert fake_model.calls == []


def test_ask_retrieves_and_cites(pipeline: RAGPipeline, fake_model, store):
    store.add_document(
        "policy", "Refunds are accepted within 30 days of purchase with a receipt.", {"name": "policy.md"}
    )
    store.add_document("other", "The office cafeteria opens at eight in the morning.", {"name": "office.md"})

    ans = pipeline.ask("How many days for refunds?", top_k=1)

    assert ans.answer == "The answer is 42 [1]."
    assert ans.model == "fake-model"
    assert (ans.prompt_tokens, ans.completion_tokens) == (100, 10)
    assert len(ans.sources) == 1 and ans.sources[0].document_id == "policy"
    assert 0.0 <= ans.sources[0].score <= 1.0

    system, user = fake_model.calls[0]
    assert system == SYSTEM_PROMPT
    assert "Refunds are accepted" in user and "cafeteria" not in user
