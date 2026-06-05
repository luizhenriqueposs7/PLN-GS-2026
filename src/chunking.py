"""
Segmentação semântica em chunks (Etapa 4 do desafio).

- Alvo de 512 a 1024 tokens por chunk (contagem com tiktoken cl100k_base, usada
  como aproximação independente do tokenizer do modelo).
- Respeita a estrutura: quebra os chunks nos títulos (Capítulo, Seção, Art. N,
  Crédito N, títulos numerados, linhas em CAIXA ALTA) para não cortar uma
  seção/artigo/crédito no meio.
- Nunca fragmenta um bloco (parágrafo, tabela ou lista); só divide um bloco
  isolado por frases quando ele sozinho passa do limite máximo.
- Gera relatório com total de chunks, distribuição por categoria e tamanho médio.
"""
from __future__ import annotations

from collections import defaultdict

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # type: ignore

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# Detecção de títulos (limites semânticos fortes)

_HEADING_RES = [
    re.compile(r"^\s*(CAP[ÍI]TULO|SE[ÇC][ÃA]O|T[ÍI]TULO|ANEXO|PARTE)\b", re.IGNORECASE),
    re.compile(r"^\s*Art(?:igo)?\.?\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*Cr[ée]dito\s+\S", re.IGNORECASE),
    re.compile(r"^\s*\d+(\.\d+){0,3}\s+\p{Lu}[^.]{0,80}$"),  # "3.2 Sistemas fotovoltaicos"
]


def _is_heading(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines or len(lines) > 2:
        return False
    first = lines[0].strip()
    if len(first) > 90:
        return False
    for rgx in _HEADING_RES:
        if rgx.match(first):
            return True
    # linha curta toda em CAIXA ALTA (provável título)
    letters = [c for c in first if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters) and not first.endswith("."):
        return True
    return False


_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=\p{Lu}|\d)")


def _split_block_by_sentences(block: str, target_max: int) -> list[str]:
    """Último recurso: divide um bloco grande por frases mantendo <= target_max."""
    sents = _SENT_SPLIT.split(block)
    pieces, cur, cur_tok = [], [], 0
    for s in sents:
        st = count_tokens(s)
        if cur and cur_tok + st > target_max:
            pieces.append(" ".join(cur))
            cur, cur_tok = [s], st
        else:
            cur.append(s)
            cur_tok += st
    if cur:
        pieces.append(" ".join(cur))
    return pieces


# Chunking

def chunk_document(
    text: str,
    meta: dict,
    *,
    target_max: int = 1024,
    target_min: int = 512,
    break_on_heading: bool = True,
) -> list[dict]:
    """Segmenta um documento já limpo em chunks com metadados herdados de `meta`.

    `meta` deve conter ao menos `doc_id`; demais chaves (titulo, fonte, categoria,
    subcategoria, ano, vigencia, url...) são copiadas para cada chunk.
    """
    doc_id = meta["doc_id"]
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]

    chunks: list[dict] = []
    cur: list[str] = []
    cur_tok = 0
    section = meta.get("titulo", doc_id)

    def flush():
        nonlocal cur, cur_tok
        if cur:
            body = "\n\n".join(cur).strip()
            if body:
                chunks.append({"text": body, "n_tokens": count_tokens(body), "section": section})
            cur, cur_tok = [], 0

    for block in blocks:
        is_head = _is_heading(block)
        if is_head:
            # título: fecha o chunk atual e atualiza a seção corrente
            if break_on_heading:
                flush()
            section = block.strip().splitlines()[0].strip()

        bt = count_tokens(block)

        if bt > target_max:
            flush()
            for piece in _split_block_by_sentences(block, target_max):
                chunks.append({"text": piece, "n_tokens": count_tokens(piece), "section": section})
            continue

        if cur and cur_tok + bt > target_max:
            flush()
        cur.append(block)
        cur_tok += bt

    flush()

    # mescla um chunk final pequeno (< target_min) com o anterior, se couber
    merged: list[dict] = []
    for ch in chunks:
        if (
            merged
            and ch["n_tokens"] < target_min
            and merged[-1]["section"] == ch["section"]
            and merged[-1]["n_tokens"] + ch["n_tokens"] <= target_max
        ):
            merged[-1]["text"] += "\n\n" + ch["text"]
            merged[-1]["n_tokens"] = count_tokens(merged[-1]["text"])
        else:
            merged.append(ch)

    # monta os dicts finais com metadados e id estável
    base = {k: v for k, v in meta.items() if k != "doc_id"}
    out: list[dict] = []
    for i, ch in enumerate(merged):
        out.append({
            "chunk_id": f"{doc_id}::{i:03d}",
            "doc_id": doc_id,
            **base,
            "section": ch["section"],
            "n_tokens": ch["n_tokens"],
            "text": ch["text"],
        })
    return out


def build_chunk_report(chunks: list[dict]) -> dict:
    """Relatório: total, distribuição por categoria/subcategoria e tokens."""
    toks = [c["n_tokens"] for c in chunks]
    by_cat: dict[str, int] = defaultdict(int)
    by_sub: dict[str, int] = defaultdict(int)
    by_doc: dict[str, int] = defaultdict(int)
    for c in chunks:
        by_cat[c.get("categoria", "?")] += 1
        by_sub[c.get("subcategoria", "?")] += 1
        by_doc[c.get("doc_id", "?")] += 1
    n = len(chunks) or 1
    return {
        "total_chunks": len(chunks),
        "tokens_medio": round(sum(toks) / n, 1),
        "tokens_min": min(toks) if toks else 0,
        "tokens_max": max(toks) if toks else 0,
        "por_categoria": dict(by_cat),
        "por_subcategoria": dict(by_sub),
        "por_documento": dict(by_doc),
    }
