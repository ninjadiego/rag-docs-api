import pytest

from app.chunking import split_text


def test_empty_text_yields_no_chunks():
    assert split_text("   \n  ") == []


def test_short_text_is_single_chunk():
    chunks = split_text("Hello world.", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert (chunks[0].start, chunks[0].end) == (0, 12)


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    # consecutive windows overlap in the source
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start < a.end
    # nothing lost: the last chunk reaches the end of the text
    assert chunks[-1].end == len(text)


def test_prefers_paragraph_boundary():
    para = "First paragraph sentence one. Sentence two.\n\nSecond paragraph starts here and keeps going for a while."
    chunks = split_text(para, chunk_size=60, overlap=10)
    assert chunks[0].text.endswith("Sentence two.")


def test_does_not_split_mid_word():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    for c in split_text(text, chunk_size=25, overlap=5):
        assert not c.text.startswith(" ")
        assert c.text.split()[0] in text.split()


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("x", size, overlap)
