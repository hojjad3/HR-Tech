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
    questions: list[MultipleChoiceQuestion] = Field(min_length=3, max_length=5)


EXAM_GEN_SYSTEM_PROMPT = """You are an Expert AI Technical Examiner.
Your task is to generate 3-5 highly tailored Multiple-Choice Questions (MCQ) for a shortlisted candidate based on their resume context and the specific AI product being built.

Each question MUST contain exactly 4 distinct options (A, B, C, D) and specify the 0-indexed position of the correct answer (0, 1, 2, or 3).

You MUST respond strictly with a valid JSON object matching this schema:
{
  "candidate_name": "string",
  "candidate_email": "string",
  "job_title": "string",
  "product_summary": "string",
  "questions": [
    {
      "question_id": 1,
      "topic": "string (e.g. Arabic RAG & Chunking Strategy)",
      "question_text": "string (Technical MCQ Question Prompt)",
      "options": [
        "Option A text",
        "Option B text",
        "Option C text",
        "Option D text"
      ],
      "correct_option_index": 0,
      "explanation": "string (Why option A is correct)"
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
) -> TechnicalExam | None:
    if not evaluation.passed:
        print(f"[EXAM GEN] Skipping exam generation for candidate '{evaluation.candidate_name}' (Passed=False).")
        return None

    print(f"[EXAM GEN] Generating tailored Multiple-Choice Exam (MCQ) for passed candidate '{evaluation.candidate_name}'...")
    client, model = _get_llm_client()

    combined_context = "\n---\n".join(resume_context_chunks) if resume_context_chunks else "No resume context."

    if client is None:
        mock_questions = [
            MultipleChoiceQuestion(
                question_id=1,
                topic="Arabic Legal RAG Chunking",
                question_text="عند بناء نظام RAG للتشريعات القانونية باللغة العربية، ما هي أفضل استراتيجية تقطيع (Chunking) لمنع ضياع السياق التشريعي؟",
                options=[
                    "التقطيع الثابت بناءً على عدد الحروف (Fixed-size character chunking 500 chars)",
                    "التقطيع الهيكلي بناءً على المواد والفقرات القانونية (Semantic Legal Structure Chunking)",
                    "تقطيع النصوص عشوائياً بناءً على علامات الترقيم فقط",
                    "عدم استخدام التقطيع وإرسال الملف كاملاً للنظام",
                ],
                correct_option_index=1,
                explanation="التقطيع الهيكلي المبني على المواد والفقرات يضمن حفظ الوحدة الموضوعية والسياق التشريعي بدون تجزئة المادة القانونية الواحدة.",
            ),
            MultipleChoiceQuestion(
                question_id=2,
                topic="Vector Retrieval Precision",
                question_text="أي من الآليات التالية تضمن أعلى دقة استرجاع (Retrieval Accuracy) للمصطلحات القانونية الدقيقة باللغة العربية؟",
                options=[
                    "استخدام Dense Embeddings فقط مع Cosine Similarity",
                    "الدمج بين البحث الدلالي واللفظي (Hybrid Search: BM25 + Dense Vector Search)",
                    "استخدام البحث النصي التقليدي SQL LIKE Query فقط",
                    "اعتماد الكلمات المفتاحية العشوائية",
                ],
                correct_option_index=1,
                explanation="الـ Hybrid Search يجمع بين دقة البحث الكلاسيكي بالكلمات المفتاحية (BM25) وقدرة الـ Dense Embeddings على فهم المعنى الدلالي.",
            ),
            MultipleChoiceQuestion(
                question_id=3,
                topic="FastEmbed & ChromaDB Integration",
                question_text="ما هي الميزة الرئيسية لاستخدام مكتبة FastEmbed مقارنة بـ PyTorch/HuggingFace في بيئات الإنتاج خفيفة الوزن؟",
                options=[
                    "تحتاج إلى كروت شاشة GPU ضخمة لتطبيق العمليات",
                    "تستعين بـ ONNX Runtime لتوليد الـ Embeddings بسرعة عالية وبدون أوفرهيد لـ PyTorch",
                    "تقوم بتخزين البيانات مباشرة في قاعدة بيانات SQL",
                    "تتطلب الاتصال الدائم بشبكة الإنترنت لتشغيل النماذج",
                ],
                correct_option_index=1,
                explanation="مكتبة FastEmbed تعتمد على ONNX Runtime مما يوفر سرعة عالية وبصمة memory خفيفة جداً بدون تحميل مكتبة PyTorch الثقيلة.",
            ),
        ]
        exam = TechnicalExam(
            candidate_name=evaluation.candidate_name,
            candidate_email=evaluation.candidate_email,
            job_title=strategy.job_title,
            product_summary=strategy.product_summary,
            questions=mock_questions,
        )
        print(f"[EXAM GEN] Successfully generated fallback Multiple-Choice Exam with {len(exam.questions)} questions.")
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

    print(f"[EXAM GEN] Successfully generated Multiple-Choice Exam for '{exam.candidate_name}' ({len(exam.questions)} MCQ questions).")
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
        product_summary="نظام مساعد قانوني ذكي (Legal AI Assistant) يعمل بتقنية RAG على النصوص العربية.",
        job_title="AI/RAG System Engineer (Arabic NLP Specialist)",
        must_have_skills=["Python", "Retrieval-Augmented Generation (RAG)", "Vector Databases"],
        evaluation_criteria=["خبرة في RAG العربية"],
    )

    exam_res = generate_technical_exam(
        evaluation=test_eval,
        strategy=test_strategy,
        resume_context_chunks=["Jane Doe has 5 years of Python & RAG development experience."],
    )

    if exam_res:
        print("\n--- Generated Multiple Choice Technical Exam JSON ---")
        print(json.dumps(exam_res.model_dump(), indent=2, ensure_ascii=False))
