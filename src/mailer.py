import sys
import resend
from src.config import settings
from src.exam_generator import TechnicalExam

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def format_exam_html(exam: TechnicalExam, google_form_url: str | None = None) -> str:
    questions_html = ""
    for idx, q in enumerate(exam.questions, start=1):
        options_html = "".join(
            [
                f"<li style='margin-bottom: 8px; color: #0f172a; font-size: 14px;'><strong style='color: #2563eb;'>{chr(65+i)}:</strong> {opt}</li>"
                for i, opt in enumerate(q.options)
            ]
        )
        questions_html += f"""
        <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-left: 5px solid #2563eb; padding: 18px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <h4 style="margin-top: 0; color: #0f172a; font-size: 16px; font-weight: 700; margin-bottom: 10px;">Q{idx}: [{q.topic}]</h4>
            <p style="font-size: 15px; color: #1e293b; font-weight: 600; margin-bottom: 14px; line-height: 1.5;">{q.question_text}</p>
            <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
                {options_html}
            </ul>
        </div>
        """

    form_button_section = ""
    if google_form_url and google_form_url.strip():
        url = google_form_url.strip()
        form_button_section = f"""
        <div style="background: #f0fdf4; border: 2px solid #86efac; padding: 20px; border-radius: 12px; text-align: center; margin: 25px 0;">
            <h3 style="color: #166534; margin: 0 0 10px 0; font-size: 18px; font-weight: 700;">📝 Online Google Form Assessment</h3>
            <p style="color: #15803d; font-size: 14px; margin-bottom: 16px;">Click the button below to complete your MCQ technical assessment directly on Google Forms:</p>
            <a href="{url}" target="_blank" style="display: inline-block; background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); color: #ffffff; text-decoration: none; padding: 16px 42px; border-radius: 8px; font-size: 17px; font-weight: 800; box-shadow: 0 4px 14px rgba(22, 163, 74, 0.4);">👉 Complete Assessment on Google Forms</a>
            <p style="color: #64748b; font-size: 12px; margin-top: 12px; margin-bottom: 0;">Direct Link: <a href="{url}" target="_blank" style="color: #2563eb;">{url}</a></p>
        </div>
        """
    else:
        form_button_section = """
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; padding: 18px; border-radius: 10px; text-align: center; margin: 25px 0;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0; font-size: 16px;">📝 Technical Multiple-Choice Assessment Instructions</h4>
            <p style="color: #1e3a8a; font-size: 14px; margin: 0;">Please review the questions below and reply directly to this email with your chosen answers (e.g. <strong>Q1: B, Q2: A, Q3: C</strong>) within 48 hours.</p>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #0f172a; max-width: 720px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 25px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="color: #ffffff; margin: 0; font-size: 24px;">Technical Assessment Invitation</h1>
            <p style="color: #cbd5e1; margin-top: 5px;">Role: {exam.job_title}</p>
        </div>
        <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-top: none; padding: 25px; border-radius: 0 0 8px 8px;">
            <p style="color: #0f172a;">Dear <strong>{exam.candidate_name}</strong>,</p>
            <p style="color: #0f172a;">Congratulations! Based on our automated screening of your experience, we are thrilled to invite you to the technical evaluation phase for the <strong>{exam.job_title}</strong> role.</p>
            
            <div style="background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 14px 18px; border-radius: 6px; margin: 20px 0;">
                <p style="margin: 0; color: #0369a1; font-size: 14px; font-weight: 600;"><strong>Project Overview:</strong> {exam.product_summary}</p>
            </div>

            {form_button_section}

            <h3 style="color: #0f172a; margin-top: 25px;">Technical Multiple-Choice Assessment Questions</h3>
            <p style="color: #334155;">Please review the tailored assessment questions below:</p>
            
            {questions_html}

            <p style="color: #0f172a; margin-top: 30px;">Best regards,<br><strong>Talent Acquisition & AI Hiring Automation Team</strong></p>
        </div>
    </body>
    </html>
    """
    return html_content


def send_candidate_exam_email(
    exam: TechnicalExam,
    sender_email: str | None = None,
    override_recipient_email: str | None = None,
    google_form_url: str | None = None,
) -> bool:
    target_recipient = override_recipient_email.strip() if override_recipient_email and override_recipient_email.strip() else exam.candidate_email
    from_address = sender_email.strip() if sender_email and sender_email.strip() else settings.SENDER_EMAIL

    print(f"[MAILER] Preparing technical assessment email for '{exam.candidate_name}' (Target Recipient: {target_recipient}, From: {from_address}, Google Form URL: {google_form_url or 'None'})...")
    html_body = format_exam_html(exam, google_form_url=google_form_url)

    if settings.RESEND_API_KEY:
        print("[MAILER] Dispatching email via Resend API...")
        resend.api_key = settings.RESEND_API_KEY
        params = {
            "from": from_address,
            "to": [target_recipient],
            "subject": f"Technical Assessment Invitation: {exam.job_title}",
            "html": html_body,
        }
        try:
            email_res = resend.Emails.send(params)
            print(f"[MAILER] Email successfully sent via Resend API to '{target_recipient}'! ID: {email_res.get('id')}")
            return True
        except Exception as e:
            print(f"[MAILER] Error sending email via Resend API: {e}")
            return False
    else:
        print("[MAILER] No Resend API Key configured. Printing HTML email preview to terminal (Dry Run Mode):")
        print("=" * 60)
        print(f"From: {from_address}")
        print(f"To: {target_recipient}")
        print(f"Subject: Technical Assessment Invitation: {exam.job_title}")
        print("-" * 60)
        print(f"Candidate: {exam.candidate_name}")
        print(f"Google Form URL: {google_form_url or 'None'}")
        print(f"Questions Count: {len(exam.questions)}")
        print("=" * 60)
        return True


if __name__ == "__main__":
    from src.exam_generator import MultipleChoiceQuestion

    sample_exam = TechnicalExam(
        candidate_name="Jane Doe",
        candidate_email="jane.doe@example.com",
        job_title="AI/RAG System Engineer (Arabic NLP Specialist)",
        product_summary="Legal AI Assistant system powered by RAG on Arabic documents.",
        questions=[
            MultipleChoiceQuestion(
                question_id=1,
                topic="Arabic RAG Chunking",
                question_text="What chunking strategy best preserves statutory context?",
                options=["Fixed character chunking", "Semantic legal clause chunking", "Random split", "No chunking"],
                correct_option_index=1,
                explanation="Preserves context",
            )
        ],
    )

    send_candidate_exam_email(sample_exam, google_form_url="https://docs.google.com/forms/d/e/sample_form_id/viewform")
