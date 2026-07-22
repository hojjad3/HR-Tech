import os
import sys
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
    pass_threshold: int,
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

    print(f"[GUI] Starting screening pipeline for {len(resume_paths)} resumes (Questions: {num_questions}, Threshold: {pass_threshold})...")

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

    # 4. Stage 3-5: Evaluation, Exam & Email Dispatch
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
            pass_threshold=pass_threshold,
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

    status_msg = f"✅ Pipeline Execution Finished! Processed {len(parsed_resumes)} resumes. Session: `{session_id or 'Unsaved'}`"

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
        return pd.DataFrame(columns=["Session ID", "Timestamp", "Candidates", "Screening Status", "Target Job Title", "Product Concept"])
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


def load_session_details_view(session_id: str):
    """Load session details for the previewer."""
    if not session_id or not session_id.strip():
        return "Enter a Session ID to preview."
    details = storage.get_session_details(session_id.strip())
    if not details:
        return f"Session '{session_id}' not found."

    strategy = details.get("strategy", {})
    candidates = details.get("candidates", [])

    html = f"""
    <div style="font-family: 'Plus Jakarta Sans', system-ui, sans-serif; max-width: 1000px; color: #f8fafc;">
        <div style="background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); padding: 18px 24px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 16px;">
            <h3 style="color: #6366f1; margin: 0 0 6px 0; font-size: 18px; font-weight: 700;">📋 Session: {details['session_id']}</h3>
            <p style="color: #94a3b8; font-size: 13px; margin: 0 0 12px 0;">Executed: {details['timestamp']}</p>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 8px; border-left: 4px solid #6366f1;">
                <p style="color: #818cf8; font-weight: 700; margin: 0; font-size: 15px;">🎯 Target Role: {strategy.get('job_title', 'N/A')}</p>
                <p style="color: #cbd5e1; font-size: 13px; margin-top: 6px; line-height: 1.5;">{strategy.get('product_summary', 'N/A')}</p>
            </div>
        </div>
        
        <h4 style="color: #f8fafc; font-size: 16px; margin: 16px 0 10px 0;">Evaluated Candidates ({len(candidates)})</h4>
    """
    for c in candidates:
        status_color = "#10b981" if c["passed"] else "#f43f5e"
        status_text = "PASSED ✅" if c["passed"] else "FAILED ❌"
        html += f"""
        <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 10px; margin-bottom: 10px; border: 1px solid rgba(255, 255, 255, 0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #f8fafc; font-weight: 700; font-size: 15px;">{c['candidate_name']}</span>
                <span style="color: {status_color}; font-weight: 800; font-size: 14px; background: rgba(0,0,0,0.3); padding: 4px 12px; border-radius: 20px;">{c['match_score']}/100 — {status_text}</span>
            </div>
            <p style="color: #64748b; font-size: 13px; margin: 4px 0 8px 0;">✉️ {c['candidate_email']}</p>
            <p style="color: #cbd5e1; font-size: 13px; margin: 0; line-height: 1.5; background: rgba(30, 41, 59, 0.4); padding: 10px; border-radius: 6px;">{c['reasoning']}</p>
        </div>
        """
    html += "</div>"
    return html


# -----------------------------------------------------------------------------
# ULTRA-SLEEK MODERN PROFESSIONAL CSS
# -----------------------------------------------------------------------------
MODERN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}

body, .gradio-container {
    background-color: #090d16 !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 16px 24px !important;
}

/* Header Banner */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 500;
    margin: 0;
}

/* Tabs Styling */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    margin-bottom: 20px !important;
    gap: 4px !important;
}

.tab-nav button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #94a3b8 !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
}

.tab-nav button:hover:not(.selected) {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f1f5f9 !important;
}

/* Card Containers & Blocks */
.block, .form, .group {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

/* Label styling overrides */
label span, .block-title, label {
    background: transparent !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-transform: none !important;
    padding: 0 !important;
}

/* Text Inputs & Textareas */
input[type="text"], textarea, select {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    background: rgba(30, 41, 59, 0.95) !important;
}

/* Primary Action Button */
.btn-primary, button.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

.btn-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.55) !important;
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
}

/* Secondary Button */
.btn-secondary, button.secondary {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover, button.secondary:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Tables & Dataframes */
.dataframe-container, table {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

table th {
    background: #1e293b !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}

table td {
    background: rgba(15, 23, 42, 0.5) !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}

/* File Upload Component */
.file-preview, .upload-container, .gr-box {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
}

/* Markdown Text */
.markdown-text {
    color: #cbd5e1 !important;
}

.markdown-text h3 {
    color: #f8fafc !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #6366f1 !important;
}
"""


def build_gui():
    with gr.Blocks(title="HR AI Assistant — Autonomous Screening Engine", css=MODERN_CSS) as app:
        
        # Hero Header Banner
        gr.HTML(
            """
            <div class="hero-header">
                <div>
                    <h1 class="hero-title">HR AI Assistant</h1>
                    <p class="hero-subtitle">Autonomous Resume Screening, Reverse-Engineered RAG Analysis & MCQ Exam Automation</p>
                </div>
            </div>
            """
        )

        with gr.Tabs():

            # ---------------------------------------------------------
            # TAB 1: NEW SCREENING RUN
            # ---------------------------------------------------------
            with gr.TabItem("New Screening Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_prompt_input = gr.Textbox(
                            label="Product Concept / Technical Business Requirements (Arabic or English)",
                            placeholder="يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن...",
                            lines=5,
                            value="يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن، بحيث يرفعوا القضية والـ AI يعمل تحليلات للنسخ والأحكام القديمة ويرتبلهم ملخص واستشارات بناءً على القانون المدني.",
                        )

                        input_mode_radio = gr.Radio(
                            choices=["Drag & Drop Files", "Folder Path Input"],
                            value="Drag & Drop Files",
                            label="Resume Ingestion Source Mode",
                        )

                        file_upload_input = gr.File(
                            label="Upload Candidate Resumes (PDF / TXT / DOCX)",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".docx"],
                            visible=True,
                        )

                        folder_path_input = gr.Textbox(
                            label="Local Resume Folder Path",
                            value="./data/resumes",
                            visible=False,
                        )

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### ⚙️ Screening & Candidate Thresholds")
                            num_questions_slider = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=5,
                                step=1,
                                label="MCQ Questions Per Candidate Exam",
                            )
                            pass_threshold_slider = gr.Slider(
                                minimum=30,
                                maximum=100,
                                value=settings.PASS_THRESHOLD,
                                step=5,
                                label="Minimum Qualification Threshold (%)",
                            )

                        with gr.Group():
                            gr.Markdown("### 📧 Email Dispatch Options")
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
                                    label="Auto-Dispatch Resend Email",
                                    value=False,
                                )
                                store_results_chk = gr.Checkbox(
                                    label="Persist to SQLite Database",
                                    value=True,
                                )

                        run_btn = gr.Button("Launch AI Screening Pipeline", variant="primary", elem_classes=["btn-primary"])
                        status_output = gr.Markdown("Ready to process resumes.")

            # ---------------------------------------------------------
            # TAB 2: 🎯 REVERSE-ENGINEERED STRATEGY
            # ---------------------------------------------------------
            with gr.TabItem("🎯 Product Strategy"):
                with gr.Row():
                    job_title_out = gr.Textbox(label="Target Reverse-Engineered Role", interactive=False)
                with gr.Row():
                    product_summary_out = gr.Textbox(label="AI Product Concept Architecture", lines=3, interactive=False)
                with gr.Row():
                    must_have_out = gr.Textbox(label="Must-Have Technical Stack", lines=5, interactive=False)
                    nice_have_out = gr.Textbox(label="Nice-to-Have Advanced Skills", lines=5, interactive=False)
                with gr.Row():
                    eval_crit_out = gr.Textbox(label="Evaluation Criteria Benchmarks", lines=5, interactive=False)

            # ---------------------------------------------------------
            # TAB 3: 📊 CANDIDATE RAG MATCHING DASHBOARD
            # ---------------------------------------------------------
            with gr.TabItem("📊 Candidate Dashboard"):
                gr.Markdown("### Candidate RAG Evaluation & Qualification Results")
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
            with gr.TabItem("📝 MCQ Exam Viewer"):
                gr.Markdown("### Generated Technical Assessment Questions")
                exams_html_out = gr.HTML(value="No exams generated yet.")

            # ---------------------------------------------------------
            # TAB 5: 📁 HISTORICAL SESSIONS & EXPORTS
            # ---------------------------------------------------------
            with gr.TabItem("📁 History & Exports"):
                gr.Markdown("### Historical Screening Sessions")
                refresh_history_btn = gr.Button("🔄 Refresh Session History", variant="secondary", elem_classes=["btn-secondary"])
                history_dataframe = gr.Dataframe(
                    headers=["Session ID", "Timestamp", "Candidates", "Screening Status", "Target Job Title", "Product Concept"],
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 🔍 Session Details & Downloads")
                with gr.Row():
                    selected_session_id_input = gr.Textbox(
                        label="Enter Session ID",
                        placeholder="e.g. RUN_20260723_001500",
                    )
                    with gr.Column():
                        preview_session_btn = gr.Button("🔍 Preview Session Details", variant="secondary", elem_classes=["btn-secondary"])
                        export_files_btn = gr.Button("📥 Generate CSV & JSON Exports", variant="primary", elem_classes=["btn-primary"])

                session_preview_html = gr.HTML(value="Select a session above to inspect details.")

                with gr.Row():
                    csv_download_file = gr.File(label="Download CSV Report")
                    json_download_file = gr.File(label="Download JSON Snapshot")

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
                pass_threshold_slider,
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

        preview_session_btn.click(
            fn=load_session_details_view,
            inputs=[selected_session_id_input],
            outputs=[session_preview_html],
        )

    return app


if __name__ == "__main__":
    gui = build_gui()
    print("[GUI] Launching Clean HR AI Assistant Interface on http://127.0.0.1:7860 ...")
    gui.launch(server_name="127.0.0.1", server_port=7860, share=False, css=MODERN_CSS)
