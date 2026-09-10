import io


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "documents": 0}


def test_ingest_query_delete_flow(client):
    r = client.post(
        "/documents",
        json={"name": "faq.md", "text": "Support hours are 9 to 5 on weekdays.", "metadata": {"team": "cx"}},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "faq.md" and body["chunks"] == 1
    doc_id = body["document_id"]

    r = client.post("/query", json={"question": "What are the support hours?"})
    assert r.status_code == 200
    ans = r.json()
    assert ans["answer"].startswith("The answer is 42")
    assert ans["sources"][0]["document_id"] == doc_id
    assert ans["sources"][0]["metadata"] == {"name": "faq.md", "team": "cx"}
    assert ans["usage"]["total_tokens"] == 110

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/health").json()["documents"] == 0


def test_upload_txt_file(client):
    files = {"file": ("notes.txt", io.BytesIO(b"Deploys happen on Fridays."), "text/plain")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 201 and r.json()["chunks"] == 1


def test_upload_rejects_binary_type(client):
    files = {"file": ("x.pdf", io.BytesIO(b"%PDF"), "application/pdf")}
    assert client.post("/documents/upload", files=files).status_code == 415


def test_validation_errors(client):
    assert client.post("/documents", json={"name": "", "text": "x"}).status_code == 422
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/query", json={"question": "long enough?", "top_k": 0}).status_code == 422


def test_query_without_documents(client):
    r = client.post("/query", json={"question": "Is there anything?"})
    assert r.status_code == 200
    assert r.json()["model"] is None and r.json()["sources"] == []
