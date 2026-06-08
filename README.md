# Assistente Técnico para Edifícios Verdes e Net Zero de Água e Energia

Sistema de perguntas e respostas (RAG) especializado em eficiência de água e energia em edificações. Responde a perguntas técnicas citando as fontes do corpus e roda inteiramente local, sem APIs pagas.

## Visão geral

O assistente recupera os trechos mais relevantes de um corpus de documentos técnicos brasileiros e usa um modelo de linguagem local para redigir a resposta, sempre indicando a fonte de cada informação. Quando a resposta não está no corpus, o sistema informa a ausência em vez de inventar.

- Corpus: 12 documentos técnicos abertos (normas e certificações, relatórios técnicos e manuais), cobrindo os eixos de água e energia.
- Embeddings: intfloat/multilingual-e5-base (768 dimensões).
- Banco vetorial: ChromaDB (persistente, similaridade de cosseno, filtro por metadados).
- Modelo de linguagem local: Qwen2.5 3B via Ollama.

## Estrutura

```
corpus/meta/      manifesto dos documentos (fonte, categoria, ano)
src/              lógica do pipeline, um módulo por etapa
notebooks/        notebook documentado com as oito etapas
reports/          relatórios de chunking, avaliação e comparação
requirements.txt  dependências do projeto
```

Os PDFs do corpus e os dados gerados (índice e textos processados) não são versionados; veja o `.gitignore`. O corpus é baixado automaticamente pelo notebook a partir do manifesto.

## Como executar

Pré-requisitos: Python 3.13 e Ollama instalados.

1. Crie o ambiente virtual e instale as dependências:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Baixe o modelo de linguagem no Ollama:

```
ollama pull qwen2.5:3b
```

3. Abra o notebook e execute todas as células:

```
jupyter lab notebooks/assistente_edificios_verdes.ipynb
```

O notebook baixa o corpus, faz a limpeza e o chunking, gera os embeddings, monta o índice no ChromaDB e roda a avaliação, mostrando as saídas de cada etapa.

## Resultados

Na avaliação com dez perguntas técnicas, sete foram respondidas com base no corpus, com as fontes citadas, e três foram recusadas por não constarem nos documentos. A comparação com o mesmo modelo sem RAG evidenciou alucinações (definições incorretas e referências inexistentes) que o pipeline com recuperação elimina.

## Vídeo de apresentação

Link: https://youtu.be/llxQ9D8QaNE
