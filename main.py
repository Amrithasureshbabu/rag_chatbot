import uvicorn
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse
import pdfplumber
import io
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import os

# ================= APP =================
app = FastAPI()

# ================= CONFIG =================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"
EMBED_MODEL = "all-MiniLM-L6-v2"

EMBED_DIM = 384
TOP_K = 10
MAX_CHARS_PER_CHUNK = 800

# ================= GLOBALS =================
print("Loading embedding model...")
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    return emb.numpy()

index = None
chunks = []   # { text, source, page }

# ================= HOME =================
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ================= UPLOAD MULTIPLE PDFs =================
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    global index, chunks

    all_text_chunks = []
    chunks = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        contents = await file.read()

        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page_no, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue

                words = text.split()
                chunk_size = 400
                overlap = 120

                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = " ".join(words[i:i + chunk_size]).strip()
                    if not chunk_text:
                        continue

                    chunks.append({
                        "text": chunk_text,
                        "source": file.filename,
                        "page": page_no + 1
                    })

                    all_text_chunks.append(chunk_text)

    # ❌ NO TEXT FOUND
    if not all_text_chunks:
        index = None
        return {"text": "No readable text found in PDFs"}

    # ================= EMBEDDING =================
    embeddings = embedder.encode(
        all_text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)

    return {
        "text": f"Processed {len(files)} PDFs ({len(chunks)} chunks)",
        "files": list({c["source"] for c in chunks})
    }

# ================= ASK =================
# ================= GLOBALS =================
index = None
chunks = []
embedder = SentenceTransformer(EMBED_MODEL)  # ensure defined globally

# ================= ASK =================
@app.post("/ask")
async def ask(body: dict = Body(...)):
    global index, chunks

    query = body.get("q", "").strip()   # this expects "q" key from frontend

    if not query:
        return {"answer": "Please ask a question."}

    if index is None or not chunks:
        return {"answer": "Upload documents first."}

    # Convert query to embedding
    q_emb = embedder.encode(query).astype("float32")
    _, indices = index.search(np.array([q_emb]), TOP_K)

    # Build context from retrieved chunks
    context_parts = []
    sources = set()

    for i in indices[0]:
        if i < len(chunks):
            context_parts.append(
                chunks[i]["text"][:MAX_CHARS_PER_CHUNK]
            )
            sources.add(
                f'{chunks[i]["source"]} (page {chunks[i]["page"]})'
            )

    context = "\n\n".join(context_parts)

    # Keep prompt unchanged
    prompt = f"""
You are a document-based assistant.

STRICT RULES:
- Answer ONLY from the DOCUMENT below
- Use semantic meaning (similar ideas, rephrasing allowed)
- DO NOT use external knowledge
- If answer is not present, reply exactly:
  Not found in document.

DOCUMENT:
{context}

QUESTION:
{query}

Provide a detailed explanation in 6–10 lines.
"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 400
                }
            },
            timeout=120
        )

        answer = resp.json().get("response", "").strip()

        if not answer:
            answer = "Not found in document"

        return {
            "answer": answer,
            "sources": list(sources)
        }

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
