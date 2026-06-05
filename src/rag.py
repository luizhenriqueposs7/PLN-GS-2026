"""
Pipeline RAG com LLM local via Ollama (Etapas 6 e 7 do desafio).

- Recupera os trechos mais relevantes no ChromaDB e monta um CONTEXTO numerado.
- O prompt de sistema obriga o modelo a responder SÓ com base nos trechos e a
  CITAR a fonte de cada informação (número entre colchetes).
- `answer_rag` devolve resposta + fontes citáveis; `answer_plain` chama o mesmo
  LLM sem contexto (para a comparação RAG vs LLM puro da Etapa 7).
"""
from __future__ import annotations

from .embeddings_index import query_index

DEFAULT_MODEL = "qwen2.5:3b"

SYSTEM_RAG = """Você é um assistente técnico especializado em edifícios verdes e Net Zero de água e energia.
Responda EXCLUSIVAMENTE com base nos trechos numerados fornecidos no CONTEXTO.

Regras obrigatórias:
1. Use apenas informações presentes no CONTEXTO; não recorra a conhecimento próprio.
2. Para CADA afirmação técnica, cite a fonte com o número do trecho entre colchetes, ex.: [2].
3. Se a resposta não estiver no CONTEXTO, responda exatamente: "A informação não está no corpus consultado." — e não invente nada.
4. Seja objetivo e técnico, em português. Reproduza valores, normas e parâmetros exatamente como aparecem no trecho.
"""

SYSTEM_PLAIN = """Você é um assistente técnico especializado em edifícios verdes e Net Zero de água e energia.
Responda à pergunta de forma técnica e objetiva, em português."""


def format_context(hits: list[dict]) -> str:
    blocos = []
    for i, h in enumerate(hits, 1):
        cab = f"[{i}] (Fonte: {h.get('fonte','?')} — {h.get('titulo','?')}, {h.get('ano','?')}"
        if h.get("section"):
            cab += f"; seção: {h['section']}"
        cab += ")"
        blocos.append(f"{cab}\n{h['text']}")
    return "\n\n".join(blocos)


def _chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": temperature, "num_ctx": 8192},
    )
    return resp["message"]["content"].strip()


def answer_rag(col, question: str, *, k: int = 5, where: dict | None = None,
               model: str = DEFAULT_MODEL) -> dict:
    """Recupera contexto, gera a resposta citando fontes e devolve tudo."""
    hits = query_index(col, question, k=k, where=where)
    context = format_context(hits)
    user = (
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA: {question}\n\n"
        "Responda citando as fontes pelos números entre colchetes."
    )
    answer = _chat(model, SYSTEM_RAG, user)
    sources = [{
        "n": i + 1,
        "titulo": h.get("titulo"),
        "fonte": h.get("fonte"),
        "ano": h.get("ano"),
        "vigencia": h.get("vigencia"),
        "categoria": h.get("categoria"),
        "subcategoria": h.get("subcategoria"),
        "section": h.get("section"),
        "doc_id": h.get("doc_id"),
        "score": round(float(h.get("score", 0.0)), 3),
    } for i, h in enumerate(hits)]
    return {"question": question, "answer": answer, "sources": sources, "context": context, "hits": hits}


def answer_plain(question: str, *, model: str = DEFAULT_MODEL) -> dict:
    """Mesmo LLM, SEM contexto RAG (baseline para a comparação da Etapa 7)."""
    answer = _chat(model, SYSTEM_PLAIN, question)
    return {"question": question, "answer": answer}


def ollama_ready(model: str = DEFAULT_MODEL) -> bool:
    """Confere se o Ollama está no ar e se o modelo está disponível."""
    try:
        import ollama
        tags = ollama.list()
        names = [m.get("model", m.get("name", "")) for m in tags.get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:
        return False
