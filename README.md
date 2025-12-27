Railway deployment guide

Overview
- This is a FastAPI app in `main.py` that serves an HTML frontend at `/` and endpoints `/upload` and `/ask`.

Quick local test
1. Create a virtualenv and activate it.

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Run locally with uvicorn:

   ```powershell
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

Production (Gunicorn)
- Railway expects a `web` process. Use the included `Procfile`:

   `web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app`

- Ensure `gunicorn` is installed (add it to `requirements.txt` or install manually):

   ```powershell
   pip install gunicorn
   ```

Railway deploy steps (recommended)
1. Commit the project to a Git provider (GitHub/GitLab).
2. Go to https://railway.app and create a new project.
3. Connect your Git repository and select the branch to deploy.
4. Railway will detect a Python project. In the deploy settings set the `Start` command to use the `Procfile` automatically, or set the start command explicitly:

   ```text
   gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app
   ```

5. Add environment variables in Railway project settings:
   - `OLLAMA_URL` (if using a remote Ollama API)
   - `MODEL_NAME` (optional override)

Notes & caveats
- The app includes heavy ML dependencies (`torch`, `sentence-transformers`, `faiss-cpu`). Railway limits may require a larger plan or use a Docker deployment.
- If `faiss-cpu` or `torch` fail to install on Railway, consider building a Docker image that provides prebuilt wheels or using a VM-friendly provider.
- For GPU/large models, run the model externally and set `OLLAMA_URL` to a reachable endpoint.

Files changed
- [main.py](main.py): now reads `PORT` environment variable at runtime.
- [Procfile](Procfile): added to run Gunicorn + Uvicorn worker.

If you want, I can:
- Add a `requirements-prod.txt` and update `requirements.txt` to include `gunicorn`.
- Create a `Dockerfile` suitable for Railway if you prefer container deploy.
