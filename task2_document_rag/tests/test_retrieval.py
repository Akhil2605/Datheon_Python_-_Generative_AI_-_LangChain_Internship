"""Unit tests for FAISS vector indexing and similarity retrieval."""

from task2_document_rag.document_loader import DocumentChunk
from task2_document_rag.main import EmbeddingEngine, FAISSVectorStore


def test_faiss_vector_store_indexing_and_search():
    engine = EmbeddingEngine(provider="local")
    store = FAISSVectorStore(engine)

    chunks = [
        DocumentChunk(content="The server features dual 32-core ARM CPUs.", source="server.pdf", page=1),
        DocumentChunk(content="Employees get 20 days paid time off per calendar year.", source="handbook.txt", page=1),
        DocumentChunk(content="Operating temperature range is -20 C to 55 C.", source="server.pdf", page=2),
    ]

    store.add_chunks(chunks)
    assert store.index.ntotal == 3

    # Search for server CPU
    results = store.search("processor and CPU specs", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert "ARM CPUs" in top_chunk.content
    assert top_chunk.source == "server.pdf"
    assert top_chunk.page == 1

    # Search for vacation policy
    pto_results = store.search("paid vacation days", top_k=1)
    assert len(pto_results) == 1
    assert "paid time off" in pto_results[0][0].content
    assert pto_results[0][0].source == "handbook.txt"


def test_faiss_vector_store_empty_search():
    engine = EmbeddingEngine(provider="local")
    store = FAISSVectorStore(engine)
    results = store.search("any query", top_k=3)
    assert results == []
