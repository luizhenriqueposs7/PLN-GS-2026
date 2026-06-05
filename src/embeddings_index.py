"""
Embeddings + indexação vetorial no ChromaDB (Etapa 5 do desafio).

- Modelo de embedding: intfloat/multilingual-e5-base (open-source, multilíngue,
  bom em português técnico, leve o suficiente para rodar em CPU). O e5 espera os
  prefixos "query:" e "passage:", tratados aqui automaticamente.
- ChromaDB com persistência em disco e métrica de cosseno; os metadados de cada
  chunk (categoria, subcategoria, fonte, ano, vigência, seção...) são gravados
  junto ao vetor para permitir filtro por categoria na busca.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "edificios_verdes"

_model = None


def get_embedder(model_name: str = EMBED_MODEL):
    """Carrega (uma vez) o SentenceTransformer."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    return _model


def embed_passages(texts: Iterable[str], model_name: str = EMBED_MODEL, batch_size: int = 16):
    model = get_embedder(model_name)
    inputs = [f"passage: {t}" for t in texts]
    return model.encode(
        inputs, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )


def embed_query(text: str, model_name: str = EMBED_MODEL):
    model = get_embedder(model_name)
    return model.encode(
        [f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True
    )[0]


# ChromaDB

_META_KEYS = ["doc_id", "titulo", "fonte", "categoria", "subcategoria", "ano", "vigencia", "url", "section"]


def _clean_meta(chunk: dict) -> dict:
    """ChromaDB só aceita str/int/float/bool nos metadados (sem None/listas)."""
    meta = {}
    for k in _META_KEYS:
        v = chunk.get(k, "")
        if v is None:
            v = ""
        if not isinstance(v, (str, int, float, bool)):
            v = str(v)
        meta[k] = v
    meta["n_tokens"] = int(chunk.get("n_tokens", 0))
    return meta


def get_collection(persist_dir: str | Path, collection: str = COLLECTION, reset: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection, metadata={"hnsw:space": "cosine"}
    )


def build_index(chunks: list[dict], persist_dir: str | Path, *, collection: str = COLLECTION,
                model_name: str = EMBED_MODEL, reset: bool = True):
    """Gera embeddings de todos os chunks e grava no ChromaDB persistente."""
    col = get_collection(persist_dir, collection, reset=reset)
    texts = [c["text"] for c in chunks]
    embeddings = embed_passages(texts, model_name=model_name)
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[_clean_meta(c) for c in chunks],
    )
    return col


def query_index(col, question: str, *, k: int = 5, where: dict | None = None,
                model_name: str = EMBED_MODEL) -> list[dict]:
    """Busca semântica; `where` permite filtrar por metadados (ex.: categoria)."""
    qemb = embed_query(question, model_name=model_name)
    res = col.query(
        query_embeddings=[qemb.tolist()], n_results=k,
        where=where, include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, **meta, "distance": dist, "score": 1.0 - dist})
    return hits
