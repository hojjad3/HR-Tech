import os
import sys
import tempfile
import modal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Define Modal App
app = modal.App("hr-ai-assistant")

# Define Debian Slim container image with all required system & Python dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pydantic>=2.0.0",
        "pydantic-settings",
        "pdfplumber",
        "chromadb",
        "fastembed",
        "openai",
        "resend",
        "requests",
    )
    .add_local_python_source("src")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("hr-ai-secrets")] if os.getenv("MODAL_ENVIRONMENT") else [],
    timeout=600,
)
def run_pipeline_modal(user_prompt: str, resume_files: list[dict]) -> dict:
    """Serverless Modal function entrypoint executing end-to-end HR AI Pipeline.

    resume_files: list of dicts with keys 'file_name' and 'content_bytes'
    """
    from src.exam_generator import generate_technical_exam
    from src.mailer import send_candidate_exam_email
    from src.matcher import evaluate_candidate
    from src.parser import parse_resume_pdf
    from src.strategy import generate_hiring_strategy
    from src.vector_store import VectorStoreManager

    print("=" * 70)
    print("🚀 [MODAL CLOUD PIPELINE] Starting HR AI Assessment Pipeline...")
    print("=" * 70)

    # Stage 1: Strategy Generation
    strategy = generate_hiring_strategy(user_prompt)

    # Stage 2: Resume Parsing & RAG Indexing
    vs_manager = VectorStoreManager(collection_name="modal_resumes")
    parsed_resumes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for item in resume_files:
            file_name = item["file_name"]
            content_bytes = item["content_bytes"]
            temp_path = os.path.join(temp_dir, file_name)

            with open(temp_path, "wb") as f:
                f.write(content_bytes)

            parsed = parse_resume_pdf(temp_path)
            parsed_resumes.append(parsed)
            vs_manager.index_resume(parsed)

    # Stage 3-5: Matching, Exam Generation & Email Dispatch
    pipeline_results = {
        "strategy": strategy.model_dump(),
        "evaluated_candidates": [],
        "total_processed": len(parsed_resumes),
        "total_passed": 0,
    }

    for resume in parsed_resumes:
        # Retrieve candidate chunks
        chunks = vs_manager.search(
            query=f"{strategy.job_title} {' '.join(strategy.must_have_skills)}",
            n_results=3,
            candidate_email=resume.candidate_email,
        )
        context_texts = [c["content"] for c in chunks] if chunks else [resume.raw_text]

        # Candidate Evaluation
        evaluation = evaluate_candidate(
            candidate_name=resume.candidate_name,
            candidate_email=resume.candidate_email,
            resume_context_chunks=context_texts,
            strategy=strategy,
        )

        candidate_record = {
            "candidate_name": evaluation.candidate_name,
            "candidate_email": evaluation.candidate_email,
            "match_score": evaluation.match_score,
            "passed": evaluation.passed,
            "reasoning": evaluation.reasoning,
            "exam_generated": False,
            "email_dispatched": False,
        }

        if evaluation.passed:
            pipeline_results["total_passed"] += 1
            # Dynamic Exam Generation
            exam = generate_technical_exam(
                evaluation=evaluation,
                strategy=strategy,
                resume_context_chunks=context_texts,
            )
            if exam:
                candidate_record["exam_generated"] = True
                candidate_record["exam"] = exam.model_dump()
                # Email Dispatch
                dispatched = send_candidate_exam_email(exam)
                candidate_record["email_dispatched"] = dispatched

        pipeline_results["evaluated_candidates"].append(candidate_record)

    print("=" * 70)
    print(f"✅ [MODAL CLOUD PIPELINE] Completed! Processed: {pipeline_results['total_processed']}, Shortlisted: {pipeline_results['total_passed']}")
    print("=" * 70)
    return pipeline_results


@app.local_entrypoint()
def main():
    sample_prompt = "يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن، بحيث يرفعوا القضية والـ AI يعمل تحليلات للنسخ والأحكام القديمة ويرتبلهم ملخص واستشارات بناءً على القانون المدني."

    sample_resume_content = """
    Jane Doe
    Email: jane.doe@example.com
    Summary:
    Senior AI & RAG Systems Engineer with 5 years experience in Python, Arabic NLP, Vector Databases (ChromaDB), FastAPI, and LLM fine-tuning.
    Built automated legal document analysis systems.
    """.encode("utf-8")

    resume_files = [
        {"file_name": "jane_doe_resume.pdf", "content_bytes": sample_resume_content}
    ]

    print("[MODAL LOCAL ENTRYPOINT] Triggering remote Modal Pipeline...")
    res = run_pipeline_modal.remote(sample_prompt, resume_files)
    print("\n--- Modal Execution Output Summary ---")
    print(f"Target Role: {res['strategy']['job_title']}")
    print(f"Processed: {res['total_processed']}, Passed: {res['total_passed']}")


if __name__ == "__main__":
    print("[MODAL APP] Modal Serverless Module defined successfully.")
