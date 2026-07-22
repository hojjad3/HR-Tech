import json
import sys
from pydantic import BaseModel, Field
from src.config import settings, get_llm_client
from src.strategy import HiringStrategy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class CandidateEvaluation(BaseModel):
    candidate_name: str
    candidate_email: str
    match_score: int = Field(ge=0, le=100, description="Match percentage score between 0 and 100")
    passed: bool = Field(description="True if match_score >= threshold, else False")
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


def evaluate_candidate(
    candidate_name: str,
    candidate_email: str,
    resume_context_chunks: list[str],
    strategy: HiringStrategy,
    pass_threshold: int | None = None,
) -> CandidateEvaluation:
    threshold = pass_threshold if pass_threshold is not None else settings.PASS_THRESHOLD
    print(f"[MATCHER] Evaluating candidate '{candidate_name}' ({candidate_email}) against role '{strategy.job_title}'...")

    combined_context = "\n---\n".join(resume_context_chunks) if resume_context_chunks else "No resume context available."
    client, model = get_llm_client()

    def fallback_evaluation(reason: str) -> CandidateEvaluation:
        context_lower = combined_context.lower()
        matched_skills = [skill for skill in strategy.must_have_skills if skill.lower() in context_lower]
        score = int((len(matched_skills) / max(len(strategy.must_have_skills), 1)) * 90) + 10
        score = min(max(score, 40), 95)
        passed = score >= threshold

        return CandidateEvaluation(
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            match_score=score,
            passed=passed,
            reasoning=f"Matched {len(matched_skills)}/{len(strategy.must_have_skills)} core skills: {matched_skills}. Evaluated for '{strategy.job_title}'. ({reason})",
        )

    if client is None:
        return fallback_evaluation("Offline Mode")

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

    models_to_try = [model, "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    # De-duplicate while preserving order
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    last_error = ""
    for target_model in models_to_try:
        try:
            print(f"[MATCHER] Attempting API evaluation with model '{target_model}'...")
            response = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": MATCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            data = json.loads(response.choices[0].message.content)
            data["passed"] = data.get("match_score", 0) >= threshold
            evaluation = CandidateEvaluation(**data)

            print(f"[MATCHER] Evaluated '{candidate_name}': Score={evaluation.match_score}/100, Passed={evaluation.passed}")
            return evaluation

        except Exception as e:
            last_error = str(e)
            print(f"[MATCHER] Warning: Model '{target_model}' failed or hit rate limit: {e}")
            continue

    print(f"[MATCHER] All models encountered errors ({last_error}). Using intelligent fallback scoring.")
    return fallback_evaluation(f"Rate limit fallback: {last_error[:120]}")


if __name__ == "__main__":
    test_strategy = HiringStrategy(
        product_summary="Legal AI Assistant system powered by RAG on Arabic documents.",
        job_title="AI/RAG System Engineer (Arabic NLP Specialist)",
        must_have_skills=["Python", "Retrieval-Augmented Generation (RAG)", "Vector Databases", "Arabic Natural Language Processing (NLP)"],
        nice_to_have_skills=["Hybrid Search", "FastAPI"],
        evaluation_criteria=["Experience in Arabic RAG systems"],
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
