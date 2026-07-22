import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pdfplumber
from pathlib import Path
from src.config import settings
from src.parser import parse_resume_pdf, ParsedResume
from src.vector_store import VectorStoreManager
from src.strategy import generate_hiring_strategy, HiringStrategy
from src.matcher import evaluate_candidate
from src.exam_generator import generate_technical_exam
from src.mailer import send_candidate_exam_email

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def create_sample_resumes(target_dir: str) -> list[str]:
    """Generates synthetic sample resumes if no PDF resumes are present."""
    os.makedirs(target_dir, exist_ok=True)
    
    sample_1_text = """
    Ahmad Al-Mansoor
    Email: ahmad.almansoor@example.com
    Title: AI/RAG System Engineer (Arabic NLP Specialist)
    
    Summary:
    Over 6 years of experience in Python backend development, Retrieval-Augmented Generation (RAG), and Arabic Natural Language Processing (NLP).
    Expert in Vector Databases (ChromaDB / Qdrant), FastEmbed, FastAPI, Docker, and LLM Integration & Fine-Tuning for legal document chunking strategies.
    
    Work Experience:
    - AI Engineer at LegalTech Jordan (2021-Present): Designed RAG engine for Jordanian Civil Law documents using dense vectors, LLM Integration & Fine-Tuning, and hybrid search.
    - Software Developer at TechCorp (2018-2021): Built microservices using Python, Vector Databases, and PostgreSQL.
    """

    sample_2_text = """
    Sara Smith
    Email: sara.smith@example.com
    Title: Frontend Developer
    
    Summary:
    3 years of experience in React, HTML, CSS, and Figma design. Basic exposure to Python scripts.
    Interested in learning AI engineering.
    """

    file_paths = []
    # Save as text-based fallback parsed objects for testing convenience
    txt_path_1 = os.path.join(target_dir, "ahmad_almansoor_resume.txt")
    txt_path_2 = os.path.join(target_dir, "sara_smith_resume.txt")

    with open(txt_path_1, "w", encoding="utf-8") as f:
        f.write(sample_1_text.strip())

    with open(txt_path_2, "w", encoding="utf-8") as f:
        f.write(sample_2_text.strip())

    return [txt_path_1, txt_path_2]


def run_pipeline(user_prompt: str, resume_paths: list[str]) -> None:
    print("\n" + "=" * 75)
    print("🤖 HR AI ASSISTANT: AUTOMATED RESUME SCREENING & ASSESSMENT PIPELINE")
    print("=" * 75)
    settings.print_status()
    print("-" * 75)

    # 1. Product-to-Hiring Strategy Reverse Engineering
    strategy = generate_hiring_strategy(user_prompt)

    # 2. Resume Ingestion & RAG Indexing
    print("\n[PIPELINE STAGE 2] Resume Parsing & RAG Indexing...")
    vs_manager = VectorStoreManager(collection_name="hr_pipeline_resumes")
    parsed_resumes: list[ParsedResume] = []

    for path in resume_paths:
        if path.endswith(".pdf"):
            parsed = parse_resume_pdf(path)
        else:
            # Handle text file fallback
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            from src.parser import extract_email, extract_candidate_name
            parsed = ParsedResume(
                file_name=os.path.basename(path),
                candidate_name=extract_candidate_name(content, path),
                candidate_email=extract_email(content),
                raw_text=content,
                char_count=len(content),
            )
            print(f"[PARSER] Loaded text resume '{parsed.file_name}': Candidate='{parsed.candidate_name}', Email='{parsed.candidate_email}'")

        parsed_resumes.append(parsed)
        vs_manager.index_resume(parsed)

    # 3. Candidate Evaluation, Dynamic Exam Generation & Email Dispatch
    print("\n[PIPELINE STAGE 3-5] Candidate Evaluation, MCQ Exam Generation & Email Dispatch...")
    summary_records = []

    for resume in parsed_resumes:
        print(f"\n>>> Processing Candidate: {resume.candidate_name} ({resume.candidate_email}) <<<")
        
        # RAG Search
        chunks = vs_manager.search(
            query=f"{strategy.job_title} {' '.join(strategy.must_have_skills)}",
            n_results=3,
            candidate_email=resume.candidate_email,
        )
        context_texts = [c["content"] for c in chunks] if chunks else [resume.raw_text]

        # Scoring
        evaluation = evaluate_candidate(
            candidate_name=resume.candidate_name,
            candidate_email=resume.candidate_email,
            resume_context_chunks=context_texts,
            strategy=strategy,
        )

        status_str = "SHORTLISTED (PASSED)" if evaluation.passed else "REJECTED (FAILED)"
        exam_status = "N/A"
        email_status = "N/A"

        if evaluation.passed:
            # Exam Generation
            exam = generate_technical_exam(
                evaluation=evaluation,
                strategy=strategy,
                resume_context_chunks=context_texts,
            )
            if exam:
                exam_status = f"Generated ({len(exam.questions)} MCQs)"
                # Email Dispatch
                dispatched = send_candidate_exam_email(exam)
                email_status = "Sent" if dispatched else "Failed"

        summary_records.append({
            "Name": evaluation.candidate_name,
            "Email": evaluation.candidate_email,
            "Score": f"{evaluation.match_score}/100",
            "Status": status_str,
            "Exam": exam_status,
            "Email": email_status,
        })

    # Summary Output Table
    print("\n" + "=" * 75)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Target Role: {strategy.job_title}")
    print(f"Product Concept: {strategy.product_summary}")
    print("-" * 75)
    print(f"{'Candidate Name':<22} | {'Score':<8} | {'Status':<20} | {'Exam Status':<18}")
    print("-" * 75)
    for rec in summary_records:
        print(f"{rec['Name']:<22} | {rec['Score']:<8} | {rec['Status']:<20} | {rec['Exam']:<18}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HR AI Assistant Pipeline CLI & Web GUI")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Gradio Web GUI dashboard",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن، بحيث يرفعوا القضية والـ AI يعمل تحليلات للنسخ والأحكام القديمة ويرتبلهم ملخص واستشارات بناءً على القانون المدني.",
        help="Informal hiring manager product prompt",
    )
    parser.add_argument(
        "--resumes_dir",
        type=str,
        default="./data/resumes",
        help="Directory containing PDF resumes",
    )
    args = parser.parse_args()

    if args.gui:
        from src.gui import build_gui
        print("🚀 [GUI] Launching HR AI Assistant Web Interface on http://127.0.0.1:7860 ...")
        app_gui = build_gui()
        app_gui.launch(server_name="127.0.0.1", server_port=7860, share=False)
        return

    # Always generate/refresh synthetic test files if no custom PDFs are supplied
    pdf_files = [os.path.join(args.resumes_dir, f) for f in os.listdir(args.resumes_dir)] if os.path.exists(args.resumes_dir) else []
    pdf_files = [f for f in pdf_files if f.endswith(".pdf")]

    if pdf_files:
        resume_paths = pdf_files
    else:
        print(f"[CLI] Refreshing sample test resumes in '{args.resumes_dir}'...")
        resume_paths = create_sample_resumes(args.resumes_dir)

    run_pipeline(user_prompt=args.prompt, resume_paths=resume_paths)


if __name__ == "__main__":
    main()
