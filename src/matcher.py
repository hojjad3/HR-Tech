import json
import sys
from pydantic import BaseModel, Field
from openai import OpenAI
from src.config import settings
from src.strategy import HiringStrategy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class CandidateEvaluation(BaseModel):
    candidate_name: str
    candidate_email: str
    match_score: int = Field(ge=0, le=100, description="Match percentage score between 0 and 100")
    passed: bool = Field(description="True if match_score >= 75, else False")
    reasoning: str = Field(description="Detailed evaluation reasoning highlighting strengths, gaps, and criteria matching")


MATCHER_SYSTEM_PROMPT = """You are an elite AI Technical Recruiter & Engineering Assessment Expert.
Evaluate the candidate's resume context against the provided Product Hiring Strategy.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "candidate_name": "string",
  "candidate_email": "string",
  "match_score": integer (0-100),
  "passed": boolean (true if match_score >= 75, else false),
  "reasoning": "string (detailed explanation of strengths, skill matches, gaps, and justification)"
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


def evaluate_candidate(
    candidate_name: str,
    candidate_email: str,
    resume_context_chunks: list[str],
    strategy: HiringStrategy,
    pass_threshold: int = 75,
) -> CandidateEvaluation:
    print(f"[MATCHER] Evaluating candidate '{candidate_name}' ({candidate_email}) against role '{strategy.job_title}'...")

    combined_context = "\n---\n".join(resume_context_chunks) if resume_context_chunks else "No resume context available."
    client, model = _get_llm_client()

    if client is None:
        # Fallback scoring logic for demonstration when API keys are absent
        context_lower = combined_context.lower()
        matched_skills = [skill for skill in strategy.must_have_skills if skill.lower() in context_lower]
        score = int((len(matched_skills) / max(len(strategy.must_have_skills), 1)) * 90) + 10
        score = min(max(score, 40), 95)
        passed = score >= pass_threshold

        evaluation = CandidateEvaluation(
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            match_score=score,
            passed=passed,
            reasoning=f"Matched skills: {matched_skills}. Evaluated against product summary '{strategy.product_summary}'. Passed threshold ({pass_threshold}): {passed}.",
        )
        print(f"[MATCHER] Evaluated '{candidate_name}': Score={evaluation.match_score}/100, Passed={evaluation.passed}")
        return evaluation

    user_prompt = f"""
TARGET PRODUCT & HIRING STRATEGY:
- Target Job Title: {strategy.job_title}
- Product Summary: {strategy.product_summary}
- Must-Have Skills: {json.dumps(strategy.must_have_skills, ensure_ascii=False)}
- Nice-to-Have Skills: {json.dumps(strategy.nice_to_have_skills, ensure_ascii=False)}
- Evaluation Criteria: {json.dumps(strategy.evaluation_criteria, ensure_ascii=False)}

CANDIDATE INFORMATION:
- Name: {candidate_name}
- Email: {candidate_email}

CANDIDATE RESUME CONTEXT CHUNKS:
\"\"\"
{combined_context}
\"\"\"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": MATCHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    # Ensure passed status aligns with threshold
    data["passed"] = data.get("match_score", 0) >= pass_threshold
    evaluation = CandidateEvaluation(**data)

    print(f"[MATCHER] Evaluated '{candidate_name}': Score={evaluation.match_score}/100, Passed={evaluation.passed}")
    return evaluation


if __name__ == "__main__":
    test_strategy = HiringStrategy(
        product_summary="نظام مساعد قانوني ذكي (Legal AI Assistant) يعمل بتقنية RAG على النصوص العربية.",
        job_title="AI/RAG System Engineer (Arabic NLP Specialist)",
        must_have_skills=["Python", "Retrieval-Augmented Generation (RAG)", "Vector Databases", "Arabic Natural Language Processing (NLP)"],
        nice_to_have_skills=["Hybrid Search", "FastAPI"],
        evaluation_criteria=["خبرة عملية في RAG باللغة العربية", "القدرة على التعامل مع المستندات القانونية"],
    )

    chunks = [
        "Jane Doe is a Senior AI Engineer specializing in Arabic Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG). She built Legal Tech document search engines using Python, ChromaDB, and FastAPI."
    ]

    eval_result = evaluate_candidate(
        candidate_name="Jane Doe",
        candidate_email="jane.doe@example.com",
        resume_context_chunks=chunks,
        strategy=test_strategy,
    )

    print("\n--- Candidate Evaluation Result ---")
    print(json.dumps(eval_result.model_dump(), indent=2, ensure_ascii=False))
