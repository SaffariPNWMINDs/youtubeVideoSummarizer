import chromadb
from openai import OpenAI

client = OpenAI()
chroma_client = chromadb.Client()

def is_video_indexed(video_id: str) -> bool:
    collection = chroma_client.get_or_create_collection(name=video_id)
    return len(collection.get()["ids"]) > 0


def chunk_transcript(transcript: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(transcript):
        end = start + chunk_size
        chunks.append(transcript[start:end])
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def store_embeddings(video_id: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    collection = chroma_client.get_or_create_collection(name=video_id)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{video_id}_{i}" for i in range(len(chunks))],
    )


def retrieve_relevant_chunks(video_id: str, question: str, top_k: int = 3) -> list[str]:
    question_embedding = embed_texts([question])[0]
    collection = chroma_client.get_or_create_collection(name=video_id)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )
    return results["documents"][0]
