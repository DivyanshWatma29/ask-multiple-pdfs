"""Small offline tests for the PDF pipeline (no Ollama connection required)."""
from app import get_text_chunks


def test_chunking_keeps_source():
    chunks = get_text_chunks([{"text": "Hello world. " * 200, "source": "sample.pdf", "page": 2}])
    assert chunks
    assert all(metadata["source"] == "sample.pdf" for _, metadata in chunks)
    assert all(metadata["page"] == 2 for _, metadata in chunks)
