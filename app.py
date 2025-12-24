from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse
import pdfplumber
import pytesseract
from PIL import Image
import io
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= APP =================
app = FastAPI()

# ================= CONFIG =================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"
EMBED_MODEL = "all-MiniLM-L6-v2"

EMBED_DIM = 384
TOP_K = 4
MAX_CHARS_PER_CHUNK = 400
# ================= GLOBALS =================
embedder = SentenceTransformer(EMBED_MODEL)
index = None
chunks = []

# ================= HELPERS =================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def chunk_text(text, chunk_size=500, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# ================= HOME =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Document Chatbot</title>
</head>
<body>
<h2>AI Document Chatbot — RAG</h2>

<form id="uploadForm">
    <input type="file" id="file" />
    <button type="submit">Upload & Extract</button>
</form>

<p id="status"></p>

<hr>

<input type="text" id="question" placeholder="Ask a question" style="width:300px;">
<button onclick="ask()">Ask</button>

<pre id="chat"></pre>

<script>
document.getElementById("uploadForm").onsubmit = async (e) => {
    e.preventDefault();
    let fileInput = document.getElementById("file");
    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    let res = await fetch("/upload", {
        method: "POST",
        body: formData
    });
    let data = await res.json();
    document.getElementById("status").innerText = data.text;
};

async function ask() {
    let q = document.getElementById("question").value;
    let res = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({q: q})
    });
    let data = await res.json();
    document.getElementById("chat").innerText += "\\nYou: " + q + "\\nBot: " + data.answer + "\\n";
}
</script>
</body>
</html>
"""


# ================= UPLOAD =================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global index, chunks

    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"text": "Only PDF files allowed"}, status_code=400)

    contents = await file.read()
    if not contents.startswith(b"%PDF"):
        return JSONResponse({"text": "Invalid PDF file"}, status_code=400)

    full_text = ""

    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text or text.strip() == "":
                    img = page.to_image(resolution=300)
                    text = pytesseract.image_to_string(img.original, lang="eng")
                full_text += clean_text(text) + " "
    except Exception:
        return JSONResponse({"text": "Failed to read PDF"}, status_code=400)

    if len(full_text) < 500:
        return JSONResponse({"text": "No readable text found"}, status_code=400)

    # Chunking
    raw_chunks = chunk_text(full_text)

    # Embedding
    embeddings = embedder.encode(
        raw_chunks,
        show_progress_bar=False,
        convert_to_numpy=True
    ).astype("float32")

    # FAISS (safe logic)
    chunks = raw_chunks
    if len(raw_chunks) < 100:
        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(embeddings)
    else:
        quantizer = faiss.IndexFlatL2(EMBED_DIM)
        index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, 100)
        index.train(embeddings)
        index.add(embeddings)

    return {"text": "Document processed successfully"}


# ================= ASK =================
@app.post("/ask")
async def ask(body: dict = Body(...)):
    global index, chunks

    q = body.get("q", "").strip()
    if not q:
        return JSONResponse({"answer": "Ask a question"}, status_code=400)

    if index is None or len(chunks) == 0:
        return JSONResponse({"answer": "Upload a document first"}, status_code=400)

    q_emb = embedder.encode(q).astype("float32")
    _, I = index.search(np.array([q_emb]), TOP_K)

    context_chunks = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            context_chunks.append(chunks[idx][:MAX_CHARS_PER_CHUNK])

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a document-based assistant.

Use ONLY the document below.
If the answer is not present, say: Not found in document.

DOCUMENT:
{context}

QUESTION:
{q}

Answer briefly.
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
                    "num_predict": 200
                }
            },
            timeout=60
        )
        answer = resp.json().get("response", "").strip()
        if not answer:
            answer = "Not found in document"
        return {"answer": answer}

    except requests.exceptions.Timeout:
        return JSONResponse({"answer": "Model timeout. Try again."}, status_code=504)
