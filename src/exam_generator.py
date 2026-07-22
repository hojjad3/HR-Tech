import json
import sys
from pydantic import BaseModel, Field
from openai import OpenAI
from src.config import settings
from src.matcher import CandidateEvaluation
from src.strategy import HiringStrategy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class MultipleChoiceQuestion(BaseModel):
    question_id: int
    topic: str
    question_text: str
    options: list[str] = Field(min_length=4, max_length=4, description="List of 4 multiple-choice options (A, B, C, D)")
    correct_option_index: int = Field(ge=0, le=3, description="0-indexed position of the correct answer (0 for A, 1 for B, 2 for C, 3 for D)")
    explanation: str = Field(description="Brief explanation of why the correct option is right")


class TechnicalExam(BaseModel):
    candidate_name: str
    candidate_email: str
    job_title: str
    product_summary: str
    questions: list[MultipleChoiceQuestion] = Field(min_length=1, max_length=10)


EXAM_GEN_SYSTEM_PROMPT = """You are an Expert AI Technical Examiner.
Your task is to generate highly tailored Multiple-Choice Questions (MCQ) for a shortlisted candidate based on their resume context and the specific AI product being built.

CRITICAL LANGUAGE & FORMAT REQUIREMENTS:
1. You MUST generate all questions, options, topics, and explanations STRICTLY IN ENGLISH ONLY. Do NOT use Arabic or any language other than English in the exam content.
2. Each question MUST contain exactly 4 distinct options (A, B, C, D) and specify the 0-indexed position of the correct answer (0, 1, 2, or 3).
3. Generate the EXACT number of questions requested by the user.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "candidate_name": "string",
  "candidate_email": "string",
  "job_title": "string",
  "product_summary": "string",
  "questions": [
    {
      "question_id": 1,
      "topic": "string (e.g. RAG & Vector Chunking Strategy)",
      "question_text": "string (Technical MCQ Question Prompt in English)",
      "options": [
        "Option A text in English",
        "Option B text in English",
        "Option C text in English",
        "Option D text in English"
      ],
      "correct_option_index": 0,
      "explanation": "string (Why option A is correct, in English)"
    }
  ]
}
"""


def _get_llm_client() -> tuple[OpenAI | None, str]:
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        ), settings.LLM_MODEL
    elif settings.OPENAI_API_KEY:
        return OpenAI(api_key=settings.OPENAI_API_KEY), "gpt-4o-mini"
    else:
        return None, "mock"


def generate_technical_exam(
    evaluation: CandidateEvaluation,
    strategy: HiringStrategy,
    resume_context_chunks: list[str],
    num_questions: int = 3,
) -> TechnicalExam | None:
    if not evaluation.passed:
        print(f"[EXAM GEN] Skipping exam generation for candidate '{evaluation.candidate_name}' (Passed=False).")
        return None

    print(f"[EXAM GEN] Generating {num_questions} tailored English Multiple-Choice Questions (MCQ) for candidate '{evaluation.candidate_name}'...")
    client, model = _get_llm_client()

    combined_context = "\n---\n".join(resume_context_chunks) if resume_context_chunks else "No resume context."

    if client is None:
        mock_question_pool = [
            MultipleChoiceQuestion(
                question_id=1,
                topic="Legal RAG Chunking Strategy",
                question_text="When building a RAG engine for legal documents and codes, which chunking strategy best prevents legislative context loss during vector retrieval?",
                options=[
                    "Fixed-size character chunking at 500 characters",
                    "Semantic Legal Structure Chunking based on articles and clauses",
                    "Random text splitting based solely on punctuation marks",
                    "Sending the entire un-chunked document directly into prompt context",
                ],
                correct_option_index=1,
                explanation="Semantic legal structure chunking preserves thematic cohesion and legal clause context without severing indivisible statutory articles.",
            ),
            MultipleChoiceQuestion(
                question_id=2,
                topic="Vector Retrieval Precision",
                question_text="Which of the following retrieval approaches ensures the highest search precision for precise legal terminology?",
                options=[
                    "Dense embeddings with Cosine Similarity only",
                    "Hybrid Search combining BM25 keyword matching with dense vector search",
                    "Traditional SQL LIKE text queries only",
                    "Random keyword indexing",
                ],
                correct_option_index=1,
                explanation="Hybrid Search combines exact keyword matching (BM25) for specialized legal terms with dense embeddings for semantic search.",
            ),
            MultipleChoiceQuestion(
                question_id=3,
                topic="FastEmbed & ONNX Runtime Performance",
                question_text="What is the primary architectural advantage of FastEmbed compared to PyTorch in lightweight production deployments?",
                options=[
                    "Requires high-end dedicated GPU clusters",
                    "Leverages ONNX Runtime for ultra-fast embedding inference without heavy PyTorch memory overhead",
                    "Stores data directly into relational SQL tables",
                    "Requires continuous internet connectivity to run embedding inference",
                ],
                correct_option_index=1,
                explanation="FastEmbed relies on ONNX Runtime, offering low latency and a minimal memory footprint without loading heavy PyTorch dependencies.",
            ),
            MultipleChoiceQuestion(
                question_id=4,
                topic="Arabic Text Preprocessing in RAG",
                question_text="Why is custom text normalization critical when preprocessing Arabic legal text before vector embedding?",
                options=[
                    "Arabic text cannot be embedded into vector space",
                    "Diacritics (Tashkeel) and orthographic variations can alter vector distance if not normalized",
                    "Vector databases only support ASCII characters",
                    "Normalization increases LLM context window size",
                ],
                correct_option_index=1,
                explanation="Normalizing diacritics, Alef variants, and Tatweel ensures consistent embedding representations across search queries and document chunks.",
            ),
            MultipleChoiceQuestion(
                question_id=5,
                topic="Async Vector Search & Concurrency",
                question_text="How should ChromaDB vector database queries be handled in a high-concurrency FastAPI backend?",
                options=[
                    "Block the main asyncio event loop with synchronous queries",
                    "Offload vector database operations to background worker threads or async task queues",
                    "Disable database concurrency completely",
                    "Re-instantiate vector index on every HTTP request",
                ],
                correct_option_index=1,
                explanation="Offloading heavy vector search operations to threadpools or async task queues prevents blocking the main event loop under high traffic.",
            ),
        ]

        # Return requested number of questions (up to 5 in mock mode)
        selected_questions = mock_question_pool[: min(num_questions, len(mock_question_pool))]
        for idx, q in enumerate(selected_questions, start=1):
            q.question_id = idx

        exam = TechnicalExam(
            candidate_name=evaluation.candidate_name,
            candidate_email=evaluation.candidate_email,
            job_title=strategy.job_title,
            product_summary=strategy.product_summary,
            questions=selected_questions,
        )
        print(f"[EXAM GEN] Successfully generated English Multiple-Choice Exam with {len(exam.questions)} questions.")
        return exam

    user_prompt = f"""
TARGET ROLE & PRODUCT:
- Job Title: {strategy.job_title}
- Product Summary: {strategy.product_summary}
- Must-Have Skills: {json.dumps(strategy.must_have_skills, ensure_ascii=False)}

CANDIDATE PROFILE:
- Name: {evaluation.candidate_name}
- Email: {evaluation.candidate_email}
- Match Score: {evaluation.match_score}/100
- Recruiter Reasoning: {evaluation.reasoning}

REQUESTED NUMBER OF MCQ QUESTIONS: {num_questions}

RESUME CONTEXT:
\"\"\"
{combined_context}
\"\"\"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXAM_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    exam = TechnicalExam(**data)

    print(f"[EXAM GEN] Successfully generated English Multiple-Choice Exam for '{exam.candidate_name}' ({len(exam.questions)} MCQs).")
    return exam


if __name__ == "__main__":
    test_eval = CandidateEvaluation(
        candidate_name="Jane Doe",
        candidate_email="jane.doe@example.com",
        match_score=85,
        passed=True,
        reasoning="Strong match in Python, FastAPI, and RAG architectures.",
    )
    test_strategy = HiringStrategy(
        product_summary="Legal AI Assistant system powered by RAG on Arabic legal documents.",
        job_title="AI/RAG System Engineer (Arabic NLP Specialist)",
        must_have_skills=["Python", "Retrieval-Augmented Generation (RAG)", "Vector Databases"],
        evaluation_criteria=["Experience in Arabic RAG systems"],
    )

    exam_res = generate_technical_exam(
        evaluation=test_eval,
        strategy=test_strategy,
        resume_context_chunks=["Jane Doe has 5 years of Python & RAG development experience."],
        num_questions=4,
    )

    if exam_res:
        print(f"\n--- Generated English Technical Exam ({len(exam_res.questions)} MCQs) ---")
        print(json.dumps(exam_res.model_dump(), indent=2, ensure_ascii=False))
