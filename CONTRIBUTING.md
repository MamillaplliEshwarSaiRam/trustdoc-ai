# Contributing

Thanks for your interest in improving TrustDoc AI.

## Local Setup

```bash
git clone <repo-url>
cd trustdoc-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The app works without API keys by using local TF-IDF retrieval and local extractive answers. Add a Google Gemini or OpenAI key only if you want semantic embeddings or generated answers.

## Development Guidelines

- Keep changes focused and easy to review.
- Do not commit `.env`, API keys, private documents, or generated local data.
- Prefer small, readable Python modules over large all-in-one files.
- Preserve citation behavior when changing retrieval or answer generation.
- Run a basic compile check before opening a pull request:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/trustdoc-pycache python -m py_compile app.py src/*.py
```

## Good First Issues

- Add automated RAG evaluation examples.
- Add persistent embedding cache.
- Add Docker support.
- Improve UI screenshots and demo documentation.
- Add OCR support for scanned PDFs.
