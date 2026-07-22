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
from src.google_forms import GoogleFormsManager, fetch_form_responses
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
    create_google_forms: bool,
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

    # 4. Stage 3-5: Evaluation, Exam, Google Forms & Email
    candidate_table_data = []
    candidate_records = []
    generated_exams_html = []

    # Initialize Google Forms manager if needed
    gf_manager = None
    if create_google_forms:
        gf_manager = GoogleFormsManager()
        if not gf_manager.is_available:
            print("[GUI] Google Forms not configured. Forms will not be created.")
            gf_manager = None

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
        form_url = None

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

                # Create Google Form if enabled
                if gf_manager and gf_manager.is_available:
                    form_info = gf_manager.create_exam_form(exam_obj)
                    if form_info:
                        form_url = form_info.get("responder_url")

                if auto_send_emails:
                    email_dispatched = send_candidate_exam_email(
                        exam=exam_obj,
                        sender_email=sender_email,
                        override_recipient_email=recipient_override,
                        google_form_url=form_url,
                    )

        record = {
            "candidate_name": evaluation.candidate_name,
            "candidate_email": evaluation.candidate_email,
            "match_score": evaluation.match_score,
            "passed": evaluation.passed,
            "reasoning": evaluation.reasoning,
            "exam": exam_obj.model_dump() if exam_obj else None,
            "email_dispatched": email_dispatched,
            "google_form_url": form_url,
        }
        candidate_records.append(record)

        target_email = recipient_override.strip() if recipient_override and recipient_override.strip() else evaluation.candidate_email
        candidate_table_data.append({
            "Candidate Name": evaluation.candidate_name,
            "Extracted / Target Email": target_email,
            "Match Score": evaluation.match_score,
            "Status": "PASSED ✅" if evaluation.passed else "FAILED ❌",
            "Google Form": form_url or "N/A",
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

        # Save Google Form metadata
        if gf_manager and session_id:
            for rec in candidate_records:
                if rec.get("google_form_url"):
                    # Find form_info from manager
                    form_info_data = {
                        "candidate_name": rec["candidate_name"],
                        "candidate_email": rec["candidate_email"],
                        "form_id": rec["google_form_url"].split("/d/")[1].split("/")[0] if "/d/" in (rec.get("google_form_url") or "") else "",
                        "form_url": rec.get("google_form_url", ""),
                        "responder_url": rec.get("google_form_url", ""),
                        "total_questions": len(rec["exam"]["questions"]) if rec.get("exam") else 0,
                    }
                    if form_info_data["form_id"]:
                        storage.save_google_form(session_id, form_info_data)

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
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 800px;">
        <h3 style="color: #e2e8f0; margin-bottom: 8px;">📋 Session: {details['session_id']}</h3>
        <p style="color: #94a3b8; font-size: 13px;">Created: {details['timestamp']}</p>
        <div style="background: #1e293b; padding: 14px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #3b82f6;">
            <p style="color: #93c5fd; font-weight: 600; margin: 0;">🎯 {strategy.get('job_title', 'N/A')}</p>
            <p style="color: #cbd5e1; font-size: 13px; margin-top: 6px;">{strategy.get('product_summary', 'N/A')}</p>
        </div>
        <h4 style="color: #e2e8f0;">Candidates ({len(candidates)})</h4>
    """
    for c in candidates:
        status_color = "#22c55e" if c["passed"] else "#ef4444"
        status_text = "PASSED ✅" if c["passed"] else "FAILED ❌"
        html += f"""
        <div style="background: #0f172a; padding: 12px; border-radius: 8px; margin: 8px 0; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #e2e8f0; font-weight: 600;">{c['candidate_name']}</span>
                <span style="color: {status_color}; font-weight: 700;">{c['match_score']}/100 — {status_text}</span>
            </div>
            <p style="color: #64748b; font-size: 12px; margin: 4px 0;">{c['candidate_email']}</p>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 6px;">{c['reasoning'][:200]}...</p>
        </div>
        """
    html += "</div>"
    return html


def load_google_forms_table():
    """Load all Google Forms for display."""
    forms = storage.get_all_forms()
    if not forms:
        return pd.DataFrame(columns=["Candidate", "Email", "Form URL", "Questions", "Responses", "Created", "Job Title"])
    df = pd.DataFrame(forms)
    display_cols = ["Candidate", "Email", "Form URL", "Questions", "Responses", "Created", "Job Title"]
    return df[[c for c in display_cols if c in df.columns]]


def fetch_and_store_responses(form_id: str):
    """Fetch responses from Google Forms API and store them."""
    if not form_id or not form_id.strip():
        return "⚠️ Please enter a Form ID.", pd.DataFrame()

    form_id = form_id.strip()
    responses = fetch_form_responses(form_id)

    if not responses:
        return f"No responses found for form '{form_id}'.", pd.DataFrame()

    # Store in DB
    storage.save_form_responses(form_id, responses)

    # Build display table
    table_data = []
    for r in responses:
        table_data.append({
            "Response ID": r["response_id"][:12] + "...",
            "Submitted": r["submitted_at"],
            "Score": f"{r['total_score']}/{r['max_score']}",
            "Percentage": f"{r['percentage']}%",
            "Status": "PASSED ✅" if r["passed"] else "FAILED ❌",
        })

    status_msg = f"✅ Fetched {len(responses)} response(s) for form '{form_id}'."
    return status_msg, pd.DataFrame(table_data)


def load_stored_responses(form_id: str):
    """Load stored responses from the database."""
    if not form_id or not form_id.strip():
        return pd.DataFrame(columns=["Response ID", "Submitted", "Score", "Percentage", "Status"])

    responses = storage.get_form_responses_from_db(form_id.strip())
    if not responses:
        return pd.DataFrame(columns=["Response ID", "Submitted", "Score", "Percentage", "Status"])

    table_data = []
    for r in responses:
        table_data.append({
            "Response ID": r["response_id"][:12] + "...",
            "Submitted": r["submitted_at"],
            "Score": f"{r['total_score']}/{r['max_score']}",
            "Percentage": f"{r['percentage']}%",
            "Status": "PASSED ✅" if r["passed"] else "FAILED ❌",
        })
    return pd.DataFrame(table_data)


def build_gui():
    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    .dark { background-color: #0f172a !important; }
    """

    with gr.Blocks(title="HR AI Assistant - Resume Screening & Assessment Pipeline", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as app:
        gr.Markdown(
            """
            # 🤖 HR AI Assistant
            ### Autonomous Resume Screening, RAG Analysis, Adaptive MCQ Assessment, Google Forms & Email Automation Pipeline
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
                            gr.Markdown("### ⚙️ Screening & Exam Settings")
                            with gr.Row():
                                num_questions_slider = gr.Slider(
                                    minimum=1,
                                    maximum=30,
                                    value=5,
                                    step=1,
                                    label="Number of MCQ Questions",
                                )
                                pass_threshold_slider = gr.Slider(
                                    minimum=30,
                                    maximum=100,
                                    value=settings.PASS_THRESHOLD,
                                    step=5,
                                    label="Pass Threshold (%)",
                                )

                        with gr.Group():
                            gr.Markdown("### 📧 Email & Google Forms Settings")
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
                            create_google_forms_chk = gr.Checkbox(
                                label="📝 Create Google Form for Each Candidate",
                                value=True,
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
                    headers=["Candidate Name", "Extracted / Target Email", "Match Score", "Status", "Google Form", "AI Reasoning"],
                    datatype=["str", "str", "number", "str", "str", "str"],
                    column_count=(6, "fixed"),
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
            # TAB 5: 📋 GOOGLE FORMS & RESPONSES
            # ---------------------------------------------------------
            with gr.TabItem("📋 Google Forms & Responses"):
                gr.Markdown("### 📝 Google Forms Dashboard — View Created Forms & Fetch Candidate Responses")

                refresh_forms_btn = gr.Button("🔄 Refresh Forms List", variant="secondary")
                forms_dataframe = gr.Dataframe(
                    headers=["Candidate", "Email", "Form URL", "Questions", "Responses", "Created", "Job Title"],
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 📥 Fetch Responses from Google Forms API")
                with gr.Row():
                    form_id_input = gr.Textbox(
                        label="Google Form ID",
                        placeholder="Enter the Form ID to fetch responses for",
                    )
                    fetch_responses_btn = gr.Button("📥 Fetch & Grade Responses", variant="primary")

                fetch_status_output = gr.Markdown("")
                responses_dataframe = gr.Dataframe(
                    headers=["Response ID", "Submitted", "Score", "Percentage", "Status"],
                    interactive=False,
                    wrap=True,
                )

            # ---------------------------------------------------------
            # TAB 6: 📁 SAVED RESULTS & EXPORT HISTORY
            # ---------------------------------------------------------
            with gr.TabItem("📁 Saved Results & Export History"):
                gr.Markdown("### Historical Screening Sessions (SQLite Database)")
                refresh_history_btn = gr.Button("🔄 Refresh Session History", variant="secondary")
                history_dataframe = gr.Dataframe(
                    headers=["Session ID", "Timestamp", "Target Job Title", "Product Concept"],
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### 🔍 Session Details Preview")
                with gr.Row():
                    selected_session_id_input = gr.Textbox(
                        label="Enter Session ID to Preview / Export",
                        placeholder="RUN_20260722_220000",
                    )
                    with gr.Column():
                        preview_session_btn = gr.Button("🔍 Preview Session", variant="secondary")
                        export_files_btn = gr.Button("📥 Generate Export Files (CSV & JSON)", variant="primary")

                session_preview_html = gr.HTML(value="Select a session to preview.")

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
                pass_threshold_slider,
                sender_email_input,
                recipient_email_override,
                auto_send_email_chk,
                create_google_forms_chk,
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

        # Google Forms tab handlers
        refresh_forms_btn.click(
            fn=load_google_forms_table,
            inputs=[],
            outputs=[forms_dataframe],
        )

        fetch_responses_btn.click(
            fn=fetch_and_store_responses,
            inputs=[form_id_input],
            outputs=[fetch_status_output, responses_dataframe],
        )

    return app


if __name__ == "__main__":
    gui = build_gui()
    print("[GUI] Launching HR AI Assistant Web Interface on http://127.0.0.1:7860 ...")
    gui.launch(server_name="127.0.0.1", server_port=7860, share=False)
