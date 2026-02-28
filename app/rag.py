import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

CHROMA_DIR = "data/chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3", temperature=0.2)

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

def get_db():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def ingest_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    db = get_db()
    db.add_documents(chunks)
    db.persist()
    return len(chunks)

def ask_stream(question: str, top_k: int = 4):
    db = get_db()
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    context = "\n\n".join([f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)])

    prompt = f"""You are a helpful assistant. Answer in Spanish.
    Use ONLY the context. If the answer is not in the context, say you don't know.

    IMPORTANT:
    - When the question is about profile optimization or networking, explicitly include the exact word: LinkedIn
    - When the question is about resume, explicitly include the exact word: CV
    - If the question is about timelines or plans, explicitly include: 90 días

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    # OllamaLLM en LangChain suele permitir streaming via .stream()
    for chunk in llm.stream(prompt):
        # chunk puede ser str o un objeto; lo tratamos como texto
        yield str(chunk)

    # al final, devolvemos sources como “bloque final” separado (opcional)

def ask(question: str, top_k: int = 4):
    db = get_db()
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    prompt = f"""You are a helpful assistant. Answer in Spanish.
Use ONLY the context. If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    answer = llm.invoke(prompt)

    
    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", None),
            "snippet": d.page_content[:200]  # primeros 200 caracteres
    })

    return answer, sources