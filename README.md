# Themis Analytica — AI Law Summarizer

Backend: FastAPI + Gemini for summarization, LegalBERT for extraction, optional GPT (OpenAI-compatible) for Q&A chatbot.

## Environment

Create a `.env` file in the repo root:

```ini
# Gemini (PDF/text summarization)
GEMINI_API_KEY=your_gemini_key_here

# GPT/OpenAI (optional, for Analysis tab Q&A chatbot)
OPENAI_API_KEY=your_openai_or_gpt_key_here

# Optional overrides
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini
```

## Endpoints

- `GET /health` — health flags for dependencies (transformers, LegalBERT, OCR, Tesseract, chatbot, etc.)
- `POST /analyze-text` — summarize raw text and extract fields
- `POST /analyze-pdf` — upload a PDF; uses native text extraction then OCR fallback and summarizes (Gemini when key is present)
- `POST /chatbot/ask` — Q&A over the current summary/text. Uses OpenAI key from `.env` if present; otherwise a heuristic fallback

## Frontend

Open `Login-page/home.html`. The Analysis tab now includes a simple chatbot that sends questions to `/chatbot/ask` using the current summary and source text.

### Quick setup
1) Copy `.env.example` to `.env` and paste your keys.
2) Install deps: pip install -r requirements.txt
3) Start API: uvicorn summarizer_api:app --host 127.0.0.1 --port 8000 --reload
4) Open the dashboard page and use the Analysis tab.

