"""
Gera o notebook do desafio (assistente_edificios_verdes.ipynb) com nbformat.
Rode:  python notebooks/build_notebook.py
Depois execute o .ipynb (nbconvert --execute) para popular as saídas.
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# Capa
md("""# Assistente Técnico para Edifícios Verdes e Net Zero de Água e Energia
### Sistema RAG com LLM local e citação de fontes

**Desafio:** construir, do zero, um assistente técnico especializado em **Edifícios Verdes e Net Zero de Energia e Água**, capaz de responder perguntas técnicas com precisão e **citando sempre a fonte** que embasou a resposta, tudo executável **localmente**, sem APIs externas pagas.

Este notebook percorre as 8 etapas do desafio:

| Etapa | Conteúdo |
|---|---|
| 1 | Planejamento e escopo (justificativa das tecnologias) |
| 2 | Construção do corpus (12 documentos, 3 categorias) |
| 3 | Limpeza e normalização |
| 4 | Segmentação (chunking) |
| 5 | Embeddings e indexação (ChromaDB) |
| 6 | Pipeline RAG com LLM local (Ollama) |
| 7 | Avaliação (10 perguntas + RAG vs LLM puro) |
| 8 | Relatório crítico + visualização t-SNE |

> A lógica do pipeline está em módulos reutilizáveis em `src/` (importados abaixo); o notebook documenta e executa cada etapa mostrando as saídas reais.
""")

# Etapa 1
md("""## Etapa 1: Planejamento e escopo

**Recorte temático.** O título do desafio é *"edifícios eficientes quanto a água e energia"*, e o enunciado pede que as documentações orientem como **coletar e tratar água localmente** e **gerar energia** (fotovoltaica ou outra), de modo que o edifício supra metade ou toda a sua demanda. Por isso o recorte adotado é **abrangente: Energia + Água + Certificações**, tratando as certificações como eixo transversal que conecta os dois temas. Cada documento é rotulado com a subcategoria `energia`, `agua` ou `ambos`.

**Justificativa das escolhas de tecnologia**

- **Modelo de embedding: `intfloat/multilingual-e5-base` (open-source).**
  Modelo multilíngue treinado com objetivo contrastivo, com bom desempenho em **português técnico** e dimensão de 768. É leve o suficiente para rodar em **CPU** (importante para reprodutibilidade local) e usa os prefixos `query:` / `passage:`, que separam a representação da pergunta e do trecho. Alternativas mais pesadas (e5-large, BGE-M3) dariam ganho marginal a um custo de memória/tempo maior, e ficam como melhoria futura.

- **Banco vetorial: `ChromaDB`.**
  Tem **persistência em disco**, API simples e, principalmente, **filtro nativo por metadados** (`where=`), que o enunciado pede para habilitar busca por categoria. Comparado ao FAISS (que indexa vetores mas não guarda metadados nem persiste sozinho), o ChromaDB reduz o código de cola. Usamos métrica de **cosseno** com embeddings normalizados.

- **LLM local: `Qwen2.5 3B` via `Ollama` (quantizado Q4).**
  Modelo pequeno (aceito pelo enunciado), forte em multilíngue/português e que roda em CPU/GPU modesta. O **Ollama** expõe uma API local em `127.0.0.1:11434`, dispensando qualquer serviço pago. O sistema fica 100% **auditável e offline**.

**Arquitetura do pipeline**

```
PDFs/HTML  ->  extração + limpeza  ->  chunking (512 a 1024 tokens)
                                              |
                                   embeddings (e5-base)
                                              |
                                   ChromaDB (vetor + metadados)
                                              |
   pergunta  ->  embedding  ->  busca top-k (filtro por categoria)
                                              |
                       contexto numerado  ->  LLM (Qwen2.5 3B)
                                              |
                          resposta + CITAÇÃO das fontes
```
""")

code("""import sys, json
from pathlib import Path
import pandas as pd

cwd = Path.cwd()
PROJ = cwd.parent if cwd.name == "notebooks" else cwd
sys.path.insert(0, str(PROJ))

from src import corpus_io, textproc, chunking
from src import embeddings_index as eidx
from src import rag

RAW     = PROJ / "corpus" / "raw"
META    = PROJ / "corpus" / "meta"
PROC    = PROJ / "data" / "processed"
CHROMA  = PROJ / "data" / "chroma"
REPORTS = PROJ / "reports"
for d in (RAW, META, PROC, CHROMA, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_colwidth", 60)
print("Projeto:", PROJ)
print("LLM (Ollama) pronto:", rag.ollama_ready())
print("Modelo de embedding:", eidx.EMBED_MODEL)
""")

# Etapa 2
md("""## Etapa 2: Construção do corpus

Foram reunidos **12 documentos técnicos** (acima do mínimo de 10), todos **abertos e gratuitos**, cobrindo as **3 categorias** exigidas e os **2 eixos** (energia/água):

- **Normas / guias de certificação:** Guia CBIC de conservação de água, Guia CBIC da ABNT NBR 15575, GBC Brasil Casa, INI-C (Inmetro/Portaria 309/2022).
- **Relatórios técnico-científicos:** CBCS, CNI, Atlas da Eficiência Energética (EPE), artigo sobre águas cinzas.
- **Manuais de tecnologias habilitadoras:** Manual de Engenharia FV (CRESESB), Conservação e Reúso de Água (FIESP/ANA), Reúso no setor industrial (ANA), Cadernos ANEEL de micro/minigeração.

> O texto integral da **ABNT NBR 15575** e o **guia LEED completo** são pagos; foram substituídos por equivalentes gratuitos (Guia CBIC e GBC Brasil Casa). Cada documento tem metadados: fonte, categoria, subcategoria, ano e vigência.
""")

code("""manifest = corpus_io.load_manifest(META / "manifest.json")
df = pd.DataFrame(manifest)[["id", "fonte", "categoria", "subcategoria", "ano", "vigencia"]]
print(f"Total de documentos no manifest: {len(manifest)}\\n")
display(df)

print("\\nDistribuição por categoria:")
print(df["categoria"].value_counts().to_string())
print("\\nDistribuição por subcategoria (eixo):")
print(df["subcategoria"].value_counts().to_string())
""")

code("""# Download idempotente (pula os que já estão em corpus/raw)
corpus_io.download_manifest(manifest, RAW)
""")

# Etapa 3
md("""## Etapa 3: Limpeza e normalização

Para cada documento (`src/textproc.py`):
1. **Extração** do texto (pdfplumber p/ PDF; também DOCX/HTML/TXT).
2. **Remoção de ruído**: cabeçalhos/rodapés que se repetem na maioria das páginas e números de página; a seção final de *Referências/Bibliografia* (de forma conservadora, só quando aparece no terço final).
3. **Normalização**: conserto de encoding (ftfy), Unicode (NFKC), junção de palavras hifenizadas na quebra de linha e colapso de espaços.
4. **Preservação**: blocos *estruturados* (tabelas de parâmetros, listas e **requisitos numerados**) mantêm suas quebras de linha; só a prosa tem as linhas de diagramação unidas em parágrafos, para não fragmentar requisitos normativos.
""")

code("""docs = []
for item in manifest:
    fp = next(iter(RAW.glob(item["id"] + ".*")), None)
    if fp is None:
        print("[faltando]", item["id"]); continue
    res = textproc.extract_and_clean(fp)
    for k in ("id", "titulo", "fonte", "categoria", "subcategoria", "ano", "vigencia", "url"):
        res[k] = item[k]
    (PROC / (item["id"] + ".txt")).write_text(res["text"], encoding="utf-8")
    docs.append(res)

clean_df = pd.DataFrame([{
    "id": d["id"], "paginas": d["n_pages"],
    "chars_brutos": d["n_chars_raw"], "chars_limpos": d["n_chars_clean"],
    "linhas_cabecalho_removidas": len(d["removed_headers"]),
} for d in docs])
print(f"{len(docs)} documentos processados.\\n")
display(clean_df)
""")

code("""# Amostra do texto limpo de um documento (primeiros 900 caracteres)
amostra = next(d for d in docs if d["id"] == "cresesb-manual-fv-2014")
print(f"--- {amostra['titulo']} ---\\n")
print(amostra["text"][:900])
""")

# Etapa 4
md("""## Etapa 4: Segmentação (chunking)

Segmentação semântica (`src/chunking.py`) com alvo de **512 a 1024 tokens** (contados com `tiktoken`, como aproximação independente do tokenizer do modelo). Os chunks **quebram nos títulos** (Capítulo, Seção, Art. N, Crédito N, títulos numerados, linhas em CAIXA ALTA) para não cortar uma seção/artigo/crédito ao meio, e **nunca fragmentam um bloco** (parágrafo, tabela ou lista); só dividem por frases um bloco que sozinho exceda o limite.
""")

code("""all_chunks = []
for d in docs:
    meta = {
        "doc_id": d["id"], "titulo": d["titulo"], "fonte": d["fonte"],
        "categoria": d["categoria"], "subcategoria": d["subcategoria"],
        "ano": str(d["ano"]), "vigencia": d["vigencia"], "url": d["url"],
    }
    all_chunks += chunking.chunk_document(d["text"], meta)

report = chunking.build_chunk_report(all_chunks)
(REPORTS / "chunk_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

with open(PROC / "chunks.jsonl", "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\\n")

print("RELATÓRIO DE CHUNKING")
print(json.dumps(report, ensure_ascii=False, indent=2))
""")

code("""import matplotlib.pyplot as plt
toks = [c["n_tokens"] for c in all_chunks]
plt.figure(figsize=(8, 4))
plt.hist(toks, bins=30, color="#2e7d32", alpha=0.85)
plt.axvline(512, color="red", ls="--", label="512")
plt.axvline(1024, color="red", ls="--", label="1024")
plt.title("Distribuição do tamanho dos chunks (tokens)")
plt.xlabel("tokens por chunk"); plt.ylabel("nº de chunks"); plt.legend()
plt.tight_layout(); plt.savefig(REPORTS / "hist_tokens.png", dpi=110); plt.show()
""")

code("""# Exemplo de um chunk com seus metadados
ex = all_chunks[len(all_chunks) // 2]
print("chunk_id:", ex["chunk_id"], "| tokens:", ex["n_tokens"])
print("fonte:", ex["fonte"], "| categoria:", ex["categoria"], "| seção:", ex["section"])
print("-" * 70)
print(ex["text"][:700])
""")

# Etapa 5
md("""## Etapa 5: Geração de embeddings e indexação (ChromaDB)

Geramos embeddings de **todos os chunks** com o `multilingual-e5-base` (prefixo `passage:`) e indexamos no **ChromaDB persistente** (`data/chroma`), com **métrica de cosseno** e os **metadados** de cada chunk gravados junto ao vetor, o que habilita o **filtro por categoria/subcategoria** na busca.
""")

code("""col = eidx.build_index(all_chunks, CHROMA, reset=True)
print("Chunks indexados no ChromaDB:", col.count())
""")

# Etapa 6
md("""## Etapa 6: Pipeline RAG com LLM local

A busca recupera os *top-k* trechos, que viram um **CONTEXTO numerado**. O **prompt de sistema** (abaixo) obriga o modelo a responder **somente** com base nesses trechos e a **citar a fonte** de cada informação com o número entre colchetes; se a resposta não estiver no contexto, ele deve dizer que *"a informação não está no corpus consultado"*.
""")

code("""print(rag.SYSTEM_RAG)""")

code("""out = rag.answer_rag(col, "Quais sistemas um edifício pode usar para reduzir o consumo de água potável?", k=5)
print("PERGUNTA:", out["question"], "\\n")
print("RESPOSTA:\\n", out["answer"], "\\n")
print("FONTES RECUPERADAS:")
for s in out["sources"]:
    print(f"  [{s['n']}] {s['fonte']}, {s['titulo']} ({s['ano']}) | score={s['score']} | {s['categoria']}/{s['subcategoria']}")
""")

code("""# Busca com FILTRO por metadados (só documentos do eixo 'energia')
out_f = rag.answer_rag(col, "Como funciona a compensação de energia da micro e minigeração distribuída?",
                       k=5, where={"subcategoria": "energia"})
print(out_f["answer"], "\\n")
print("Fontes (todas do eixo energia):")
for s in out_f["sources"]:
    print(f"  [{s['n']}] {s['fonte']}, {s['subcategoria']}")
""")

# Etapa 7
md("""## Etapa 7: Avaliação do sistema

**10 perguntas técnicas** sobre o corpus. Para cada uma registramos a resposta do RAG e as **fontes citadas**. Incluímos propositalmente 2 perguntas **fora da cobertura** do corpus (ex.: limiares específicos do LEED, preços de mercado) para verificar se o sistema **admite a ausência** em vez de alucinar. Isso alimenta a métrica de *cobertura* do relatório. Em seguida, comparamos **3 respostas** com o **mesmo LLM sem RAG**.
""")

code("""PERGUNTAS = [
    {"q": "Quais usos não potáveis são indicados para o reúso de águas cinzas em edificações?", "comparar": True},
    {"q": "Como funciona o sistema de compensação de energia elétrica na micro e minigeração distribuída?", "comparar": True},
    {"q": "Que parâmetros o INI-C utiliza para avaliar a eficiência energética da envoltória de edificações comerciais?", "comparar": True},
    {"q": "Quais são os principais componentes de um sistema fotovoltaico conectado à rede?", "comparar": False},
    {"q": "Quais medidas de conservação de água são recomendadas para reduzir o consumo em edificações?", "comparar": False},
    {"q": "O que a certificação GBC Brasil Casa avalia em relação ao uso da água e da energia?", "comparar": False},
    {"q": "Como o guia da CBIC trata o desempenho dos sistemas hidrossanitários segundo a NBR 15575?", "comparar": False},
    {"q": "Quais indicadores de eficiência energética o Atlas da Eficiência Energética da EPE acompanha?", "comparar": False},
    # Fora da cobertura (esperado: 'não está no corpus'):
    {"q": "Qual é o limiar exato de pontos do crédito de eficiência hídrica da certificação LEED v4.1 BD+C?", "comparar": False},
    {"q": "Qual o custo médio, em reais por kWp, de um sistema fotovoltaico residencial instalado no Brasil em 2024?", "comparar": False},
]

SENTINELA = "não está no corpus"
resultados = []
for i, p in enumerate(PERGUNTAS, 1):
    r = rag.answer_rag(col, p["q"], k=5)
    coberta = SENTINELA.lower() not in r["answer"].lower()
    resultados.append({
        "n": i, "pergunta": p["q"], "coberta": coberta,
        "resposta": r["answer"],
        "fontes": [f"[{s['n']}] {s['fonte']}, {s['titulo']}" for s in r["sources"]],
        "fonte_top": f"{r['sources'][0]['fonte']} ({r['sources'][0]['score']})" if r["sources"] else "",
    })
    print(f"\\n{'='*78}\\nQ{i}. {p['q']}\\n{'-'*78}")
    print(r["answer"])
    print("Fontes:", " ; ".join(s for s in resultados[-1]["fontes"][:3]))

cobertas = sum(1 for r in resultados if r["coberta"])
print(f"\\n\\nCOBERTURA: {cobertas}/{len(PERGUNTAS)} perguntas respondidas com base no corpus.")
(REPORTS / "avaliacao.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
""")

code("""# Tabela-resumo da avaliação
aval_df = pd.DataFrame([{
    "n": r["n"], "pergunta": r["pergunta"][:55] + "...",
    "coberta": "sim" if r["coberta"] else "não (sem cobertura)",
    "fonte principal": r["fonte_top"],
} for r in resultados])
display(aval_df)
""")

code("""# Comparação RAG vs LLM puro (sem contexto), nas 3 perguntas marcadas
comparacoes = []
for p in [x for x in PERGUNTAS if x["comparar"]]:
    com_rag = rag.answer_rag(col, p["q"], k=5)
    sem_rag = rag.answer_plain(p["q"])
    comparacoes.append({"pergunta": p["q"], "rag": com_rag["answer"],
                        "fontes_rag": [f"[{s['n']}] {s['fonte']}" for s in com_rag["sources"]],
                        "puro": sem_rag["answer"]})
    print(f"\\n{'#'*78}\\nPERGUNTA: {p['q']}")
    print(f"\\n>>> COM RAG (cita fontes):\\n{com_rag['answer']}")
    print("Fontes:", " ; ".join(comparacoes[-1]["fontes_rag"][:3]))
    print(f"\\n>>> LLM PURO (sem contexto):\\n{sem_rag['answer']}")

(REPORTS / "comparacao.json").write_text(json.dumps(comparacoes, ensure_ascii=False, indent=2), encoding="utf-8")
""")

# Etapa 8
md("""## Etapa 8: Visualização t-SNE e relatório crítico

Projeção **t-SNE** dos embeddings dos chunks em 2D, colorida por **categoria** e por **subcategoria**, para inspecionar se trechos do mesmo tipo formam **clusters** coerentes no espaço vetorial.
""")

code("""import numpy as np
from sklearn.manifold import TSNE

got = col.get(include=["embeddings", "metadatas"])
X = np.array(got["embeddings"])
cats = [m["categoria"] for m in got["metadatas"]]
subs = [m["subcategoria"] for m in got["metadatas"]]
print("Matriz de embeddings:", X.shape)

perp = min(30, max(5, X.shape[0] // 4))
emb2d = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, labels, titulo in [(axes[0], cats, "por categoria"), (axes[1], subs, "por subcategoria")]:
    for lab in sorted(set(labels)):
        m = [l == lab for l in labels]
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=12, alpha=0.6, label=lab)
    ax.set_title(f"t-SNE dos chunks, {titulo}"); ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(REPORTS / "tsne.png", dpi=110); plt.show()
""")

md("""### Relatório crítico

**1. Dificuldades na coleta do corpus.** Reunir fontes técnicas brasileiras abertas exigiu contornar vários obstáculos. (i) O texto integral da **ABNT NBR 15575** e o guia completo do **LEED** são *pagos*; foram substituídos por equivalentes gratuitos e oficiais (o Guia CBIC da NBR 15575 e o GBC Brasil Casa), preservando o rigor técnico sem violar licenças. (ii) Vários links institucionais estavam *mortos ou bloqueados*: a **INI-R** (Portaria 309/2022, residencial) retornou 404 em todas as URLs testadas e foi **removida** do corpus, que ainda mantém 12 documentos (acima do mínimo de 10) e o equilíbrio 4/4/4 entre categorias; o Guia CBIC da NBR 15575 e o Caderno ANEEL caíram (404/403 por WAF) e foram recuperados via **Internet Archive** (capturas status-200). (iii) O host do **Manual CRESESB** apresentava certificado TLS com nome incorreto, exigindo download com verificação de certificado desabilitada (sinalizado como `insecure` no manifest). Todas essas decisões ficaram registradas no campo `notas` do manifest, garantindo rastreabilidade da proveniência.

**2. Dificuldades de extração e impacto na cobertura.** A etapa mais crítica foi a **extração de texto dos PDFs**. O *Atlas da Eficiência Energética da EPE* (14,8 MB, 83 páginas) produziu apenas **859 caracteres** de texto limpo e **um único chunk**: trata-se de um relatório fortemente **imagético/vetorial**, cujos gráficos e tabelas são renderizados como imagem e, portanto, invisíveis ao `pdfplumber`. O relatório do **CBCS** também rendeu pouco (13 KB, 7 chunks). A consequência aparece diretamente na avaliação: a pergunta Q8, sobre os indicadores do Atlas, foi respondida com *"a informação não está no corpus consultado"*. Mas, neste caso, isso ocorreu **não por escolha de escopo, e sim por falha de extração**: o conteúdo existe na fonte, porém nunca chegou ao índice. É uma limitação honesta do pipeline atual. A distribuição de chunks por documento ficou, assim, bastante **desigual** (CRESESB 351, GBC 280, NBR 15575 186, contra CBCS 7 e EPE 1), num total de **1.331 chunks** com média de **733 tokens**, dentro do alvo de 512 a 1024. O `tokens_max` de **3.091** revela um trade-off deliberado: blocos atômicos (parágrafos longos e, sobretudo, **tabelas normativas**) que excedem o limite **não são subdivididos**, para não fragmentar requisitos: preferiu-se preservar a integridade semântica a uniformizar o tamanho.

**3. Qualidade dos clusters (t-SNE).** A projeção por **subcategoria** é o resultado mais eloquente: os chunks de **energia** ocupam a metade esquerda, os de **água** a metade direita e os de tema **misto** o centro, uma separação temática limpa que **valida a escolha do `multilingual-e5-base`**: o espaço vetorial capturou o eixo água/energia mesmo em português técnico, sem treino específico. Já a projeção por **categoria** é mais difusa: os manuais formam um bloco coeso (puxado pelo volumoso manual FV), mas normas e relatórios se sobrepõem, porque o **tipo documental cruza o tema**: uma norma e um relatório sobre água são vizinhos semânticos ainda que de categorias distintas. Ou seja, o modelo agrupa por **assunto**, não por **gênero do documento**, comportamento esperado e desejável para a recuperação.

**4. Proporção de cobertura.** Das 10 perguntas, **7 foram respondidas com base no corpus** e 3 declinadas. Duas declinações eram **propositais** (limiar do crédito hídrico do LEED v4.1 e custo de mercado em R$/kWp): informação fora do escopo documental, e o sistema **admitiu a ausência em vez de alucinar**, exatamente o comportamento desejado. A terceira (Q8) é o gap de extração já discutido. As 7 respostas cobertas citaram **fontes corretas e pertinentes** (águas cinzas: FIESP/ANA mais artigo UNESC; micro/minigeração: ANEEL; FV: CRESESB; INI-C: INMETRO; conservação de água: CBIC; GBC Casa: GBC Brasil), com *scores* de similaridade entre **0,86 e 0,89**.

**5. Impacto do RAG (vs. LLM puro).** A comparação com o mesmo Qwen2.5 3B **sem contexto** foi reveladora. Sem RAG, o modelo **alucinou** sistematicamente: definiu "águas cinzas" erroneamente como *água de chuva* e inventou usos (geração de energia térmica); criou categorias regulatórias inexistentes ("Compensação Direta" vs. "Indireta, Tarifa de Energia Solar"); e, no caso mais grave, **errou o próprio significado da sigla INI-C** ("Índice Nacional de Eficiência Energética"), afirmando que "não é termo técnico usado no Brasil", quando na verdade é a *Instrução Normativa Inmetro*. Com RAG, as mesmas perguntas produziram valores, normas e siglas **corretos e rastreáveis** (RedCgTT, ASHRAE Standard 140, REN ANEEL 482/2012, NBR 13.969/97), cada afirmação ancorada a um trecho `[n]`. O ganho em **precisão, rastreabilidade e ausência de alucinação** é qualitativamente evidente: o RAG não só acerta mais, como permite **auditar** a origem de cada afirmação.

**6. Melhorias futuras.** (a) **OCR** (Tesseract / `ocrmypdf`) na etapa de extração, para resgatar documentos imagéticos como o Atlas da EPE e eliminar o maior gap de cobertura atual. (b) **Reranking** com *cross-encoder* e/ou embeddings maiores (e5-large, BGE-M3) para refinar a recuperação em perguntas cujo *top-k* misturou guias semelhantes (Q7, sobre a NBR 15575, trouxe trechos de dois guias CBIC distintos). (c) **Subdivisão de blocos** acima do limite e um *parser* de tabelas dedicado, reduzindo os chunks de cerca de 3 mil tokens sem quebrar requisitos. (d) **Avaliação automática** de *groundedness/faithfulness* sobre um conjunto maior de perguntas e um leve ajuste do prompt para padronizar o formato das citações `[n]` (a Q6 usou números de seção como marcadores, um desvio cosmético do padrão).
""")

# Conclusão
md("""## Conclusão

O sistema entrega um assistente técnico **local, auditável e com citação de fontes**, atendendo às 8 etapas do desafio. A recuperação por similaridade com filtro por metadados, somada a um LLM local com prompt restritivo, garante respostas rastreáveis e a recusa explícita quando a informação não consta no corpus.
""")

# Demonstração interativa
md("""## Demonstração interativa

Para consultar o assistente com qualquer pergunta, edite o texto de `pergunta` na célula abaixo e execute (Shift+Enter). A resposta vem com as fontes citadas; perguntas fora do escopo do corpus recebem a mensagem padrão de ausência.
""")

code("""pergunta = "Quais medidas de conservação de água são recomendadas para reduzir o consumo em edificações?"

resp = rag.answer_rag(col, pergunta, k=5)
print("PERGUNTA:", pergunta, "\\n")
print("RESPOSTA:\\n", resp["answer"], "\\n")
print("FONTES:")
for s in resp["sources"]:
    print(f"  [{s['n']}] {s['fonte']}, {s['titulo']} ({s['ano']})")
""")

if __name__ == "__main__":
    nb["cells"] = cells
    out = Path(__file__).parent / "assistente_edificios_verdes.ipynb"
    nbf.write(nb, str(out))
    print("Notebook gerado:", out)
    print("Células:", len(cells))
