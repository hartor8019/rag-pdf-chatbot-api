# RAG PDF Chatbot API (FastAPI + Ollama + Chroma)

API que ingiere PDFs, crea embeddings y responde preguntas usando RAG.
Incluye streaming y evaluación automática (golden set).

## Features
- `/ingest`: ingesta de PDF y creación de índice vectorial
- `/ask`: QA con fuentes
- `/ask/stream`: respuesta en streaming (token-by-token)
- Evaluación automática: `eval/run_eval.py`

## Run with Docker (recommended)
```bash
docker compose up --build