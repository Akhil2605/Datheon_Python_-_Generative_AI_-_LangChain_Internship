"""Task 2: Retrieval-Augmented Generation (RAG) Pipeline.

Loads PDF and text documents, splits text using LangChain's RecursiveCharacterTextSplitter,
stores embeddings in a FAISS vector index, retrieves relevant chunks,
and generates grounded answers with source filename and PDF page citations.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task2_document_rag.document_loader import DocumentChunk, load_document, split_documents

# Load environment variables from root .env
load_dotenv(dotenv_path=ROOT_DIR / ".env")

INSUFFICIENT_INFO_MESSAGE = "I do not have sufficient information in the provided documents to answer this question."


@dataclass
class Citation:
    source: str
    page: Optional[int]
    snippet: str

    def format(self) -> str:
        page_str = f", Page {self.page}" if self.page is not None else ""
        return f"[Source: {self.source}{page_str}]"


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    has_sufficient_context: bool = True
    retrieved_chunks: List[DocumentChunk] = field(default_factory=list)


class EmbeddingEngine:
    """Generates normalized vector embeddings using Gemini, OpenAI, or a deterministic local model."""

    def __init__(self, provider: Optional[str] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if provider:
            self.provider = provider
        elif self.gemini_key and self.gemini_key.strip() != "your_gemini_api_key_here":
            self.provider = "gemini"
        elif self.openai_key and self.openai_key.strip() != "your_openai_api_key_here":
            self.provider = "openai"
        else:
            self.provider = "local"

        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.gemini_key)
            self.model = "text-embedding-004"
            self.dimension = 768
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=self.openai_key)
            self.model = "text-embedding-3-small"
            self.dimension = 1536
        else:
            # Deterministic local TF-IDF / hash embedding model for testing and offline execution
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(n_features=512, norm=None, alternate_sign=False)
            self.dimension = 512

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings into an L2-normalized float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.provider == "gemini":
            vectors = []
            for t in texts:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=t
                )
                vectors.append(resp.embedding.values)
            arr = np.array(vectors, dtype=np.float32)

        elif self.provider == "openai":
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            arr = np.array([item.embedding for item in resp.data], dtype=np.float32)

        else:
            # Local deterministic embedding
            sparse_mat = self.vectorizer.transform(texts)
            arr = sparse_mat.toarray().astype(np.float32)

        # L2-normalize vectors so Inner Product equals Cosine Similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


class FAISSVectorStore:
    """Manages chunk indexing and similarity retrieval using FAISS."""

    def __init__(self, embedding_engine: EmbeddingEngine):
        import faiss
        self.faiss = faiss
        self.embedding_engine = embedding_engine
        self.dimension = embedding_engine.dimension
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        texts = [c.content for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Returns top_k (chunk, similarity_score) sorted by cosine similarity."""
        if not self.chunks or self.index.ntotal == 0:
            return []

        query_vec = self.embedding_engine.embed_query(query)
        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results


class RAGPipeline:
    """Orchestrates document loading, indexing, retrieval, and grounded response generation."""

    def __init__(self, provider: Optional[str] = None):
        self.embedding_engine = EmbeddingEngine(provider=provider)
        self.vector_store = FAISSVectorStore(self.embedding_engine)
        self.llm_provider = self._detect_llm_provider(provider)

    @staticmethod
    def _detect_llm_provider(forced: Optional[str]) -> str:
        if forced and forced in ["gemini", "openai"]:
            return forced
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if gemini_key and gemini_key.strip() != "your_gemini_api_key_here":
            return "gemini"
        if openai_key and openai_key.strip() != "your_openai_api_key_here":
            return "openai"
        return "offline"

    def load_and_index(self, target: Union[str, Path], chunk_size: int = 400, chunk_overlap: int = 80) -> int:
        """Loads a file or all supported documents in a directory into the FAISS index."""
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        files_to_load = []
        if path.is_dir():
            for ext in [".pdf", ".txt", ".md"]:
                files_to_load.extend(list(path.glob(f"*{ext}")))
        else:
            files_to_load.append(path)

        if not files_to_load:
            raise ValueError(f"No supported documents (.pdf, .txt, .md) found in {path}")

        total_chunks = 0
        for f in files_to_load:
            try:
                raw_docs = load_document(f)
                chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                self.vector_store.add_chunks(chunks)
                total_chunks += len(chunks)
            except Exception as exc:
                print(f"Warning: Skipping {f.name}: {exc}", file=sys.stderr)

        if total_chunks == 0:
            raise ValueError("No text chunks could be extracted from the specified documents.")

        return total_chunks

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Invokes Gemini or OpenAI to synthesize a grounded answer."""
        if self.llm_provider == "gemini":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0
                )
            )
            return resp.text.strip() if resp.text else INSUFFICIENT_INFO_MESSAGE

        elif self.llm_provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()

        raise RuntimeError("No LLM provider configured.")

    def query(self, question: str, top_k: int = 3, similarity_threshold: float = 0.05) -> RAGResponse:
        """Retrieves relevant chunks and generates a grounded citation-backed answer."""
        if not question or not question.strip():
            raise ValueError("Query string cannot be empty.")

        matches = self.vector_store.search(question, top_k=top_k)
        # Filter by threshold
        valid_matches = [m for m in matches if m[1] >= similarity_threshold]

        if not valid_matches:
            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_INFO_MESSAGE,
                citations=[],
                has_sufficient_context=False,
                retrieved_chunks=[]
            )

        retrieved_chunks = [m[0] for m in valid_matches]

        # Build citations
        seen = set()
        citations: List[Citation] = []
        for ch in retrieved_chunks:
            key = (ch.source, ch.page)
            if key not in seen:
                seen.add(key)
                snippet = (ch.content[:90] + "...") if len(ch.content) > 90 else ch.content
                citations.append(Citation(source=ch.source, page=ch.page, snippet=snippet))

        # Check if the query is asking about something completely absent
        # Build context prompt
        context_blocks = []
        for idx, ch in enumerate(retrieved_chunks, start=1):
            page_info = f", Page {ch.page}" if ch.page is not None else ""
            context_blocks.append(f"--- Context {idx} [Source: {ch.source}{page_info}] ---\n{ch.content}")

        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an accurate, grounded document question-answering assistant.\n"
            "Rules:\n"
            "1. Answer the question using ONLY the facts provided in the Context below.\n"
            "2. Whenever mentioning a fact, cite the source using the exact format: [Source: filename, Page: X] or [Source: filename].\n"
            f"3. If the Context does NOT contain sufficient information to answer the question, output EXACTLY:\n"
            f"'{INSUFFICIENT_INFO_MESSAGE}'\n"
            "4. NEVER extrapolate, assume, or bring in outside information."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question.strip()}\n\nGrounded Answer:"

        if self.llm_provider != "offline":
            try:
                answer = self._call_llm(user_prompt, system_prompt)
                is_sufficient = INSUFFICIENT_INFO_MESSAGE not in answer
                return RAGResponse(
                    question=question,
                    answer=answer,
                    citations=citations if is_sufficient else [],
                    has_sufficient_context=is_sufficient,
                    retrieved_chunks=retrieved_chunks
                )
            except Exception as exc:
                print(f"Warning: LLM generation error ({exc}). Falling back to grounded retrieval summary.", file=sys.stderr)

        # Offline / deterministic grounded synthesis fallback
        # Combines the relevant retrieved passages with citation
        passages = "\n\n".join(ch.content for ch in retrieved_chunks)
        offline_answer = (
            f"Based on {citations[0].format()}:\n"
            f"{passages}"
        )
        return RAGResponse(
            question=question,
            answer=offline_answer,
            citations=citations,
            has_sufficient_context=True,
            retrieved_chunks=retrieved_chunks
        )


def main():
    parser = argparse.ArgumentParser(description="Document RAG Pipeline (PDF & Text QA)")
    parser.add_argument("--docs", "-d", default="task2_document_rag/sample_documents", help="Path to document file or directory")
    parser.add_argument("--query", "-q", help="Question to ask about the documents")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--provider", choices=["gemini", "openai", "local"], help="Override embedding/LLM provider")

    args = parser.parse_args()

    pipeline = RAGPipeline(provider=args.provider)
    print(f"Loading and indexing documents from: {args.docs} ...")
    try:
        count = pipeline.load_and_index(args.docs)
        print(f"Indexed {count} text chunks successfully into FAISS.")
    except Exception as exc:
        print(f"Initialization Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.query:
        resp = pipeline.query(args.query, top_k=args.top_k)
        print("\n" + "=" * 60)
        print(f"Question: {resp.question}")
        print(f"Answer:\n{resp.answer}")
        if resp.citations:
            print("\nCitations:")
            for c in resp.citations:
                print(f"  - {c.format()}: {c.snippet}")
        print("=" * 60)
    else:
        print("\nEnter questions interactively (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_q = input("\nQuestion: ").strip()
                if not user_q or user_q.lower() in ["exit", "quit"]:
                    break
                resp = pipeline.query(user_q, top_k=args.top_k)
                print(f"\nAnswer:\n{resp.answer}")
                if resp.citations:
                    print("\nCitations:")
                    for c in resp.citations:
                        print(f"  - {c.format()}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
