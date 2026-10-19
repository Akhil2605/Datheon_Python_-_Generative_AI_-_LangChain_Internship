"""Unit tests for text chunking using LangChain's RecursiveCharacterTextSplitter."""

from task2_document_rag.document_loader import DocumentChunk, split_documents


def test_split_documents_basic():
    text = (
        "Paragraph 1 discusses section A of the protocol. It is quite detailed and contains multiple sentences.\n\n"
        "Paragraph 2 covers section B of the protocol, explaining how the parameters are configured.\n\n"
        "Paragraph 3 concludes the protocol with validation criteria."
    )
    doc = DocumentChunk(content=text, source="protocol.pdf", page=3)
    chunks = split_documents([doc], chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 1
    for ch in chunks:
        assert ch.source == "protocol.pdf"
        assert ch.page == 3
        assert len(ch.content) > 0


def test_split_documents_empty():
    assert split_documents([]) == []


def test_chunk_index_ordering():
    doc1 = DocumentChunk(content="First doc text segment here.", source="doc1.txt", page=1)
    doc2 = DocumentChunk(content="Second doc text segment here.", source="doc2.txt", page=2)

    chunks = split_documents([doc1, doc2], chunk_size=50, chunk_overlap=10)
    for i, ch in enumerate(chunks):
        assert ch.chunk_index == i
