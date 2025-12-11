git clone https://github.com/<your-username>/RetainIQ.git
cd RetainIQ

2.Create virtualenv:
python -m venv venv
# Windows
venv\Scripts\activate
# mac/linux
source venv/bin/activate

3.Install dependencies:
pip install -r requirements.txt

4.Set OpenAI key (example .env or environment variable):
set OPENAI_API_KEY=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
# or create .env with OPENAI_API_KEY=...

5.Run:
uvicorn src.main:app --reload
# or uvicorn main:app --reload if main.py at root

Open http://127.0.0.1:8000.

Files to check
src/main.py — FastAPI backend entrypoint
src/templates/ — Jinja2 HTML templates
docs/retainiq_prototype_documentation.pdf — design & project doc

Notes
Do NOT commit .env or secrets.
Use the .gitignore to keep venv and DB files out of repo.

