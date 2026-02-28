import os
from fastapi import HTTPException
from fastapi import FastAPI, UploadFile, File
from app.schemas import AskRequest, AskResponse
from app.rag import ingest_pdf, ask
from fastapi.responses import StreamingResponse

app = FastAPI(title="RAG PDF Chatbot API", version="1.0.0")

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        path = os.path.join("data/uploads", file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())

        n_chunks = ingest_pdf(path)
        return {"file": file.filename, "chunks_indexed": n_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask_q(req: AskRequest):
    answer, sources = ask(req.question, req.top_k)
    return {"answer": answer, "sources": sources}

from app.rag import ask_stream

@app.post("/ask/stream")
def ask_stream_q(req: AskRequest):
    def event_generator():
        for token in ask_stream(req.question, req.top_k):
            yield token
    return StreamingResponse(event_generator(), media_type="text/plain")