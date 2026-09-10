from app.store import VectorStore, document_id_for


def test_document_id_is_stable():
    assert document_id_for("a.md") == document_id_for("a.md")
    assert document_id_for("a.md") != document_id_for("b.md")


def test_add_query_delete_roundtrip(store: VectorStore):
    n = store.add_document("doc1", "Go is a compiled language. " * 20, {"name": "go.md"})
    assert n >= 1
    store.add_document("doc2", "Python is an interpreted language. " * 20, {"name": "py.md"})
    assert store.count_documents() == 2

    hits = store.query("interpreted language Python", top_k=2)
    assert hits and hits[0].document_id == "doc2"
    assert hits[0].metadata["name"] == "py.md"

    store.delete_document("doc2")
    assert store.count_documents() == 1
    assert all(h.document_id != "doc2" for h in store.query("Python", top_k=5))


def test_reingest_replaces_instead_of_duplicating(store: VectorStore):
    store.add_document("doc1", "version one " * 50)
    first = store.collection.count()
    store.add_document("doc1", "version two " * 50)
    assert store.collection.count() == first
    assert all("two" in h.text for h in store.query("version", top_k=10))


def test_query_on_empty_store(store: VectorStore):
    assert store.query("anything") == []
