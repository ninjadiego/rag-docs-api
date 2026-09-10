"""Text chunking.

Splits a document into overlapping windows so that each chunk fits comfortably
in an embedding model and a sentence cut at a window boundary still appears
whole in the neighbouring chunk.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    start: int  # character offset in the source document
    end: int


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split `text` into chunks of at most `chunk_size` characters.

    Chunks prefer to end on a paragraph, sentence or word boundary near the end
    of the window, so we do not cut words in half. Consecutive
    chunks overlap by roughly `overlap` characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _snap_to_boundary(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, index=index, start=start, end=end))
            index += 1
        if end >= len(text):
            break
        start = _snap_start(text, max(end - overlap, start + 1), end)
    return chunks


def _snap_start(text: str, start: int, end: int) -> int:
    """Advance `start` to the next word boundary so overlaps never begin mid-word."""
    if start > 0 and not text[start - 1].isspace():
        nxt = text.find(" ", start, end)
        if nxt != -1:
            return nxt + 1
    return start


def _snap_to_boundary(text: str, start: int, end: int) -> int:
    """Move `end` back to the nearest natural break.

    Paragraph breaks are worth a bigger sacrifice of window size than sentence
    ends, which in turn beat plain spaces. Each tier only looks at the tail of
    the window so chunks never collapse to a few characters.
    """
    span = end - start
    tiers = (
        (0.5, ("\n\n",)),
        (0.7, ("\n", ". ", "! ", "? ")),
        (0.85, (" ",)),
    )
    for fraction, seps in tiers:
        floor = start + int(span * fraction)
        best = -1
        best_sep = ""
        for sep in seps:
            pos = text.rfind(sep, floor, end)
            if pos > best:
                best, best_sep = pos, sep
        if best != -1:
            return best + len(best_sep)
    return end
