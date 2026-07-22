import os
import sys
import json
import tempfile
import pandas as pd
import gradio as gr

# Ensure project root is in sys.path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.parser import parse_resume_pdf, ParsedResume, extract_email, extract_candidate_name
from src.vector_store import VectorStoreManager
from src.strategy import generate_hiring_strategy, HiringStrategy
from src.matcher import evaluate_candidate, CandidateEvaluation
from src.exam_generator import generate_technical_exam, TechnicalExam
from src.mailer import send_candidate_exam_email, format_exam_html
from src import storage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def process_screening(
    user_prompt: str,
    input_mode: str,
    uploaded_files: list | None,
    folder_path: str | None,
    num_questions: int,
    sender_email: str,
    recipient_override: str,
    auto_send_emails: bool,
    store_results: bool,
):
    if not user_prompt.strip():
        return (
            "⚠️ Please enter a product concept or hiring prompt.",
            "N/A", "N/A", "N/A", "N/A", "N/A",
            pd.DataFrame(),
            "No candidate exam generated.",
            None, None
        )

    # 1. Resolve Resume Files
    resume_paths = []

    if input_mode == "Drag & Drop Files":
        if not uploaded_files:
            return (
                "⚠️ Please upload at least one resume file (PDF or TXT).",
                "N/A", "N/A", "N/A", "N/A", "N/A",
                pd.DataFrame(),
                "No candidate exam generated.",
                None, None
            )
        for f in uploaded_files:
            resume_paths.append(f.name if hasattr(f, "name") else str(f))

    elif input_mode == "Folder Path Input":
        if not folder_path or not os.path.exists(folder_path):
            return (
                f"⚠️ Folder path '{folder_path}' does not exist.",
                "N/A", "N/A", "N/A", "N/A", "N/A",
                pd.DataFrame(),
                "No candidate exam generated.",
                None, None
            )
        for fname in os.listdir(folder_path):
            if fname.endswith(".pdf") or fname.endswith(".txt"):
                resume_paths.append(os.path.join(folder_path, fname))

    if not resume_paths:
        return (
            "⚠️ No PDF or TXT resume files found.",
            "N/A", "N/A", "N/A", "N/A", "N/A",
            pd.DataFrame(),
            "No candidate exam generated.",
            None, None
        )

    print(f"[GUI] Starting screening pipeline for {len(resume_paths)} resumes (Questions per exam: {num_questions})...")

    # 2. Stage 1: Product-to-Hiring Strategy
    strategy = generate_hiring_strategy(user_prompt)

    # 3. Stage 2: Ingestion & Vector Indexing
    vs_manager = VectorStoreManager(collection_name="gui_resumes")
    parsed_resumes = []

    for path in resume_paths:
        if path.endswith(".pdf"):
            parsed = parse_resume_pdf(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            parsed = ParsedResume(
                file_name=os.path.basename(path),
                candidate_name=extract_candidate_name(content, path),
                candidate_email=extract_email(content),
                raw_text=content,
                char_count=len(content),
            )
        parsed_resumes.append(parsed)
        vs_manager.index_resume(parsed)

    # 4. Stage 3-5: Evaluation, Exam & Email
    candidate_table_data = []
    candidate_records = []
    generated_exams_html = []

    for resume in parsed_resumes:
        # Search RAG
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

        exam_obj = None
        email_dispatched = False

        if evaluation.passed:
            # Generate English MCQ exam with custom question count
            exam_obj = generate_technical_exam(
                evaluation=evaluation,
                strategy=strategy,
                resume_context_chunks=context_texts,
                num_questions=int(num_questions),
            )
            if exam_obj:
                exam_html = format_exam_html(exam_obj)
                generated_exams_html.append(exam_html)

                if auto_send_emails:
                    email_dispatched = send_candidate_exam_email(
                        exam=exam_obj,
                        sender_email=sender_email,
                        override_recipient_email=recipient_override,
                    )

        record = {
            "candidate_name": evaluation.candidate_name,
            "candidate_email": evaluation.candidate_email,
            "match_score": evaluation.match_score,
            "passed": evaluation.passed,
            "reasoning": evaluation.reasoning,
            "exam": exam_obj.model_dump() if exam_obj else None,
            "email_dispatched": email_dispatched,
        }
        candidate_records.append(record)

        target_email = recipient_override.strip() if recipient_override and recipient_override.strip() else evaluation.candidate_email
        candidate_table_data.append({
            "Candidate Name": evaluation.candidate_name,
            "Extracted / Target Email": target_email,
            "Match Score": evaluation.match_score,
            "Status": "PASSED ✅" if evaluation.passed else "FAILED ❌",
            "AI Reasoning": evaluation.reasoning,
        })

    # 5. Persistent Storage
    session_id = None
    if store_results:
        session_id = storage.save_screening_session(
            prompt=user_prompt,
            strategy_dict=strategy.model_dump(),
            candidate_records=candidate_records,
        )

    status_msg = f"✅ Pipeline Completed! Processed {len(parsed_resumes)} candidates. Session ID: {session_id or 'Not Stored'}"

    must_have_str = "\n".join([f"• {s}" for s in strategy.must_have_skills])
    nice_have_str = "\n".join([f"• {s}" for s in strategy.nice_to_have_skills])
    eval_crit_str = "\n".join([f"• {s}" for s in strategy.evaluation_criteria])

    df = pd.DataFrame(candidate_table_data)
    exams_rendered = "<hr>".join(generated_exams_html) if generated_exams_html else "No candidate passed the threshold for exam generation."

    csv_path = storage.export_session_to_csv(session_id) if session_id else None
    json_path = storage.export_session_to_json(session_id) if session_id else None

    return (
        status_msg,
        strategy.job_title,
        strategy.product_summary,
        must_have_str,
        nice_have_str,
        eval_crit_str,
        df,
        exams_rendered,
        csv_path,
        json_path,
    )


def load_history_table():
    sessions = storage.get_all_sessions()
    if not sessions:
        return pd.DataFrame(columns=["Session ID", "Timestamp", "Target Job Title", "Product Concept"])
    return pd.DataFrame(sessions)


def load_selected_session_files(session_id: str):
    if not session_id or not session_id.strip():
        return None, None
    try:
        csv_path = storage.export_session_to_csv(session_id.strip())
        json_path = storage.export_session_to_json(session_id.strip())
        return csv_path, json_path
    except Exception as e:
        print(f"[GUI ERROR] Export failed: {e}")
        return None, None


def build_gui():
    with gr.Blocks(title="HR AI Assistant - Resume Screening & Assessment Pipeline") as app:
        gr.Markdown(
            """
            # 🤖 HR AI Assistant
            ### Autonomous Resume Screening, RAG Analysis, Adaptive MCQ Assessment & Email Automation Pipeline
            """
        )

        with gr.Tabs():

            # ---------------------------------------------------------
            # TAB 1: 🚀 NEW SCREENING RUN
            # ---------------------------------------------------------
            with gr.TabItem("🚀 New Screening Run"):
                with gr.Row():
                    with gr.Column(scale=2):
                        user_prompt_input = gr.Textbox(
                            label="Product Concept / Business Request (Arabic or English)",
                            placeholder="يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن...",
                            lines=4,
                            value="يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن، بحيث يرفعوا القضية والـ AI يعمل تحليلات للنسخ والأحكام القديمة ويرتبلهم ملخص واستشارات بناءً على القانون المدني.",
                        )

                        input_mode_radio = gr.Radio(
                            choices=["Drag & Drop Files", "Folder Path Input"],
                            value="Drag & Drop Files",
                            label="Resume Input Mode",
                        )

                        file_upload_input = gr.File(
                            label="Upload Resume PDFs / TXT files",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".docx"],
                            visible=True,
                        )

                        folder_path_input = gr.Textbox(
                            label="Local Resume Folder Path",
                            value="./data/resumes",
                            visible=False,
                        )

                        with gr.Group():
                            gr.Markdown("### ⚙️ Exam & Email Dispatch Settings")
                            num_questions_slider = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=3,
                                step=1,
                                label="Number of MCQ Questions in Exam",
                            )
                            with gr.Row():
                                sender_email_input = gr.Textbox(
                                    label="Sender Email Address",
                                    value="onboarding@resend.dev",
                                    placeholder="onboarding@resend.dev",
                                )
                                recipient_email_override = gr.Textbox(
                                    label="Override Recipient Email (Optional)",
                                    placeholder="Leave blank to use candidate's extracted email",
                                )

                        with gr.Row():
                            auto_send_email_chk = gr.Checkbox(
                                label="Auto-Send MCQ Exam Email via Resend API",
                                value=False,
                            )
                            store_results_chk = gr.Checkbox(
                                label="Save Results to Persistent SQLite DB",
                                value=True,
                            )

                        run_btn = gr.Button("🚀 Run AI Screening Pipeline", variant="primary", size="lg")
                        status_output = gr.Markdown("Ready for execution.")

            # ---------------------------------------------------------
            # TAB 2: 🎯 PRODUCT STRATEGY & TECH REQUIREMENTS
            # ---------------------------------------------------------
            with gr.TabItem("🎯 Product Strategy"):
                with gr.Row():
                    job_title_out = gr.Textbox(label="Target Job Role", interactive=False)
                with gr.Row():
                    product_summary_out = gr.Textbox(label="Product Concept Architecture", lines=3, interactive=False)
                with gr.Row():
                    must_have_out = gr.Textbox(label="Must-Have Technical Skills", lines=5, interactive=False)
                    nice_have_out = gr.Textbox(label="Nice-to-Have Skills", lines=5, interactive=False)
                with gr.Row():
                    eval_crit_out = gr.Textbox(label="Evaluation Criteria Benchmarks", lines=5, interactive=False)

            # ---------------------------------------------------------
            # TAB 3: 📊 CANDIDATE MATCHING DASHBOARD
            # ---------------------------------------------------------
            with gr.TabItem("📊 Candidate Matching Dashboard"):
                gr.Markdown("### Candidate RAG Evaluation & Target Emails")
                candidates_dataframe = gr.Dataframe(
                    headers=["Candidate Name", "Extracted / Target Email", "Match Score", "Status", "AI Reasoning"],
                    datatype=["str", "str", "number", "str", "str"],
                    column_count=(5, "fixed"),
                    interactive=True,
                    wrap=True,
                )

            # ---------------------------------------------------------
            # TAB 4: 📝 MCQ ASSESSMENT VIEWER
            # ---------------------------------------------------------
            with gr.TabItem("📝 MCQ Assessment Viewer"):
                gr.Markdown("### Generated Multiple-Choice Technical Exams (English Only)")
                exams_html_out = gr.HTML(value="No exam generated yet.")

            # ---------------------------------------------------------
            # TAB 5: 📁 SAVED RESULTS & EXPORT HISTORY
            # ---------------------------------------------------------
            with gr.TabItem("📁 Saved Results & Export History"):
                gr.Markdown("### Historical Screening Sessions (SQLite Database)")
                refresh_history_btn = gr.Button("🔄 Refresh Session History", variant="secondary")
                history_dataframe = gr.Dataframe(
                    headers=["Session ID", "Timestamp", "Target Job Title", "Product Concept"],
                    interactive=False,
                )

                with gr.Row():
                    selected_session_id_input = gr.Textbox(
                        label="Enter Session ID to Export",
                        placeholder="RUN_20260722_220000",
                    )
                    export_files_btn = gr.Button("📥 Generate Export Files (CSV & JSON)", variant="primary")

                with gr.Row():
                    csv_download_file = gr.File(label="Download CSV Export")
                    json_download_file = gr.File(label="Download JSON Export")

        # ---------------------------------------------------------
        # EVENT HANDLERS & CALLBACKS
        # ---------------------------------------------------------
        def toggle_input_mode(mode):
            if mode == "Drag & Drop Files":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        input_mode_radio.change(
            fn=toggle_input_mode,
            inputs=[input_mode_radio],
            outputs=[file_upload_input, folder_path_input],
        )

        run_btn.click(
            fn=process_screening,
            inputs=[
                user_prompt_input,
                input_mode_radio,
                file_upload_input,
                folder_path_input,
                num_questions_slider,
                sender_email_input,
                recipient_email_override,
                auto_send_email_chk,
                store_results_chk,
            ],
            outputs=[
                status_output,
                job_title_out,
                product_summary_out,
                must_have_out,
                nice_have_out,
                eval_crit_out,
                candidates_dataframe,
                exams_html_out,
                csv_download_file,
                json_download_file,
            ],
        )

        refresh_history_btn.click(
            fn=load_history_table,
            inputs=[],
            outputs=[history_dataframe],
        )

        export_files_btn.click(
            fn=load_selected_session_files,
            inputs=[selected_session_id_input],
            outputs=[csv_download_file, json_download_file],
        )

    return app


if __name__ == "__main__":
    gui = build_gui()
    print("[GUI] Launching HR AI Assistant Web Interface on http://127.0.0.1:7860 ...")
    gui.launch(server_name="127.0.0.1", server_port=7860, share=False)
