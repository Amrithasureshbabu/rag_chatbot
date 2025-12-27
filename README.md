Railway deployment guide

Overview
- This is a FastAPI app in main.py that serves an HTML frontend at / and endpoints /upload and /ask.

Quick local test
1. Create a virtualenv and activate it.

   Windows (PowerShell):
   `powershell
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   pip install -r requirements.txt
   `

2. Run locally with uvicorn:

   `powershell
   uvicorn main:app --host 0.0.0.0 --port 8080
   `

Production (Gunicorn)
- Railway expects a web process. Use the included Procfile:

   web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app

Docker (recommended for heavy ML deps)
- Build the Docker image locally:

   `powershell
   docker build -t text_app:latest .
   `

- Run the container (maps port 8080):

   `powershell
   docker run -e OLLAMA_URL= http://localhost:11434/api/generate -p 8080:8080 text_app:latest
   `

Railway deploy steps (recommended)
1. Commit the project to a Git provider (GitHub/GitLab).
2. Go to https://railway.app and create a new project.
3. Connect your Git repository and select the branch to deploy.
4. In Railway, choose Docker deployment (recommended) or let Railway build from the repo. If Railway builds from source, ensure available memory and build time are sufficient.
5. Add environment variables in Railway project settings:
   - OLLAMA_URL (if using a remote Ollama API)
   - MODEL_NAME (optional override)

Notes ;& caveats
- The app includes heavy ML dependencies (	orch, sentence-transformers, aiss-cpu). Building these in Railway's default environment may fail; using the Dockerfile is recommended so you can control build steps and use prebuilt wheels.
- If you rely on a local Ollama instance, make the model reachable from Railway or host Ollama externally.

Files changed
- main.py: reads PORT from environment
- Procfile: run Gunicorn + Uvicorn worker
- Dockerfile: added for containerized deployment
- .dockerignore: added
- requirements.txt: cleaned and includes gunicorn

