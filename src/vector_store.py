import os
from typing import Any
import chromadb
from fastembed import TextEmbedding
from src.config import settings
from src.parser import ParsedResume


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


class VectorStoreManager:
    def __init__(self, collection_name: str = "resumes"):
        print(f"[VECTOR STORE] Initializing FastEmbed model '{settings.EMBEDDING_MODEL}'...")
        self.embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)

        persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)
        print(f"[VECTOR STORE] Initializing ChromaDB persistent client at '{persist_dir}'...")
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings_generator = self.embedding_model.embed(texts)
        return [e.tolist() for e in embeddings_generator]

    def index_resume(self, resume: ParsedResume) -> int:
        chunks = chunk_text(resume.raw_text)
        if not chunks:
            print(f"[VECTOR STORE] Warning: No text chunks generated for '{resume.file_name}'")
            return 0

        print(f"[VECTOR STORE] Generating embeddings for {len(chunks)} chunks of candidate '{resume.candidate_name}'...")
        embeddings = self._embed_texts(chunks)

        ids = [f"{resume.candidate_email}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "candidate_name": resume.candidate_name,
                "candidate_email": resume.candidate_email,
                "file_name": resume.file_name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        print(f"[VECTOR STORE] Successfully indexed {len(chunks)} chunks for candidate '{resume.candidate_name}' ({resume.candidate_email})")
        return len(chunks)

    def search(self, query: str, n_results: int = 3, candidate_email: str | None = None) -> list[dict[str, Any]]:
        print(f"[VECTOR STORE] Performing similarity search for query: '{query}'")
        query_embeddings = self._embed_texts([query])

        where_clause = {"candidate_email": candidate_email} if candidate_email else None

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where_clause,
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "similarity_distance": round(dist, 4),
                    }
                )

        print(f"[VECTOR STORE] Retrieved {len(formatted_results)} matching chunks.")
        return formatted_results

    def clear_collection(self) -> None:
        self.chroma_client.delete_collection(self.collection.name)
        print("[VECTOR STORE] Collection cleared.")


if __name__ == "__main__":
    sample_resume = ParsedResume(
        file_name="john_doe_resume.pdf",
        candidate_name="John Doe",
        candidate_email="john.doe@example.com",
        raw_text="""
        John Doe is a Senior Python Developer with 6 years of experience building scalable backend microservices,
        REST APIs with FastAPI, RAG engines using FastEmbed and ChromaDB, PostgreSQL database optimization,
        and Docker containerization. He led a team of 4 engineers at TechCorp.
        """,
        char_count=320,
    )

    vs = VectorStoreManager(collection_name="test_resumes")
    vs.index_resume(sample_resume)

    results = vs.search(query="Python FastAPI RAG database", n_results=2)
    for idx, res in enumerate(results, start=1):
        print(f"\n--- Result #{idx} ---")
        print(f"Candidate: {res['metadata'].get('candidate_name')}")
        print(f"Content: {res['content']}")
        print(f"Distance: {res['similarity_distance']}")
