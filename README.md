# HR AI Assistant 🚀

A production-ready, fully automated HR Resume Screening & Assessment Pipeline written in Pure Python. Powered by lightweight RAG (`fastembed` + `chromadb`), Pydantic validation, Modal serverless deployment, and Resend email automation.

---

## 🏗 System Architecture

1. **Intent Analysis & Strategy Generation**: Converts informal hiring manager prompts into structured JSON evaluation criteria using an LLM.
2. **Resume Parsing & Lightweight RAG**: Extracts text and emails from PDF resumes (`pdfplumber` + Regex), embeds text chunks locally with `fastembed` (BGE-small ONNX model), and indexes into ChromaDB.
3. **Candidate Matching & Scoring**: Performs semantic similarity retrieval and structured scoring against job criteria.
4. **Dynamic Exam Generation**: Auto-generates adaptive 3–5 question technical exams targeting candidate skill gaps.
5. **Modal Cloud Backend**: Wraps the pipeline into serverless functions deployed on Modal (`modal.App`).
6. **Automated Email Dispatch**: Formats and emails personalized technical exams to passing candidates via Resend API.

---

## 📦 Installation & Setup

Using [`uv`](https://github.com/astral-sh/uv) package manager:

```bash
# Clone the repository
git clone https://github.com/hojjad3/HR-Tech.git
cd HR-Tech

# Install dependencies
uv sync

# Copy environment variables template
cp .env.example .env
# Fill in your API keys in .env
```

---

## ⚡ Usage

```bash
# Run CLI Application locally
uv run python src/app.py

# Deploy to Modal Cloud
uv run modal deploy src/modal_app.py
```
