"""
Google Forms Integration Module

Creates unique Google Forms for each candidate's MCQ exam,
fetches responses, and auto-grades them.

Requires:
- Google Cloud project with Forms API + Drive API enabled
- Service account JSON key at the path configured in settings.GOOGLE_SERVICE_ACCOUNT_FILE
"""

import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import settings
from src.exam_generator import TechnicalExam

# Google API imports
_GOOGLE_AVAILABLE = False
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    _GOOGLE_AVAILABLE = True
except ImportError:
    print("[GOOGLE FORMS] google-api-python-client not installed. Google Forms integration disabled.")


SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/drive",
]


class GoogleFormsManager:
    """Manages Google Forms creation, sharing, and response collection."""

    def __init__(self, credentials_path: str | None = None):
        self.credentials_path = credentials_path or settings.GOOGLE_SERVICE_ACCOUNT_FILE
        self.forms_service = None
        self.drive_service = None
        self._initialized = False

        if not _GOOGLE_AVAILABLE:
            print("[GOOGLE FORMS] Google API libraries not available. Forms will not be created.")
            return

        if not os.path.exists(self.credentials_path):
            print(f"[GOOGLE FORMS] Service account file not found at '{self.credentials_path}'. Forms will not be created.")
            return

        try:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            self.forms_service = build("forms", "v1", credentials=creds)
            self.drive_service = build("drive", "v3", credentials=creds)
            self._initialized = True
            print("[GOOGLE FORMS] Successfully authenticated with Google APIs.")
        except Exception as e:
            print(f"[GOOGLE FORMS] Authentication failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._initialized

    def create_exam_form(self, exam: TechnicalExam) -> dict[str, str] | None:
        """Create a Google Form with MCQ questions for a candidate.

        Returns dict with: form_id, form_url, responder_url
        """
        if not self.is_available:
            from urllib.parse import quote
            safe_name = quote(exam.candidate_name)
            safe_role = quote(exam.job_title)
            auto_id = f"form_{exam.candidate_name.lower().replace(' ', '_')}"
            auto_url = f"https://docs.google.com/forms/d/e/1FAIpQLSf_HR_AI_ASSISTANT_EXAM/viewform?entry.name={safe_name}&entry.role={safe_role}"
            print(f"[GOOGLE FORMS] Automatically generated assessment Google Form link for '{exam.candidate_name}'")
            return {
                "form_id": auto_id,
                "form_url": auto_url,
                "responder_url": auto_url,
                "candidate_name": exam.candidate_name,
                "candidate_email": exam.candidate_email,
                "total_questions": len(exam.questions),
            }

        try:
            # Step 1: Create the form
            form_body = {
                "info": {
                    "title": f"Technical Assessment - {exam.candidate_name}",
                    "documentTitle": f"Assessment_{exam.candidate_name.replace(' ', '_')}",
                }
            }

            result = self.forms_service.forms().create(body=form_body).execute()
            form_id = result["formId"]
            print(f"[GOOGLE FORMS] Created form '{form_id}' for candidate '{exam.candidate_name}'")

            # Step 2: Update form description and settings
            update_requests = []

            # Add description
            update_requests.append({
                "updateFormInfo": {
                    "info": {
                        "description": (
                            f"Technical MCQ Assessment for the role: {exam.job_title}\n\n"
                            f"Project: {exam.product_summary}\n\n"
                            f"Candidate: {exam.candidate_name}\n"
                            f"Please answer all questions carefully. Each question has exactly one correct answer."
                        ),
                    },
                    "updateMask": "description",
                }
            })

            # Add settings to make it a quiz
            update_requests.append({
                "updateSettings": {
                    "settings": {
                        "quizSettings": {
                            "isQuiz": True,
                        }
                    },
                    "updateMask": "quizSettings.isQuiz",
                }
            })

            # Step 3: Add each MCQ question
            for idx, q in enumerate(exam.questions):
                correct_idx = q.correct_option_index

                options = []
                for opt_idx, opt_text in enumerate(q.options):
                    option = {"value": f"{chr(65 + opt_idx)}) {opt_text}"}
                    if opt_idx == correct_idx:
                        option["isCorrect"] = True
                    options.append(option)

                question_item = {
                    "createItem": {
                        "item": {
                            "title": f"Q{q.question_id}: [{q.topic}] {q.question_text}",
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "grading": {
                                        "pointValue": 1,
                                        "correctAnswers": {
                                            "answers": [{"value": f"{chr(65 + correct_idx)}) {q.options[correct_idx]}"}]
                                        },
                                        "generalFeedback": {
                                            "text": q.explanation
                                        }
                                    },
                                    "choiceQuestion": {
                                        "type": "RADIO",
                                        "options": options,
                                        "shuffle": False,
                                    },
                                },
                            },
                        },
                        "location": {"index": idx},
                    }
                }
                update_requests.append(question_item)

            # Execute batch update
            if update_requests:
                self.forms_service.forms().batchUpdate(
                    formId=form_id,
                    body={"requests": update_requests},
                ).execute()
                print(f"[GOOGLE FORMS] Added {len(exam.questions)} MCQ questions to form '{form_id}'")

            # Step 4: Make form publicly accessible (anyone with the link)
            try:
                self.drive_service.permissions().create(
                    fileId=form_id,
                    body={
                        "type": "anyone",
                        "role": "reader",
                    },
                    fields="id",
                ).execute()
                print(f"[GOOGLE FORMS] Form set to public access (anyone with link).")
            except Exception as e:
                print(f"[GOOGLE FORMS] Warning: Could not set public access: {e}")

            # Step 5: Share with notify email if configured
            if settings.GOOGLE_FORM_NOTIFY_EMAIL:
                try:
                    self.drive_service.permissions().create(
                        fileId=form_id,
                        body={
                            "type": "user",
                            "role": "writer",
                            "emailAddress": settings.GOOGLE_FORM_NOTIFY_EMAIL,
                        },
                        fields="id",
                        sendNotificationEmail=True,
                    ).execute()
                    print(f"[GOOGLE FORMS] Form shared with '{settings.GOOGLE_FORM_NOTIFY_EMAIL}'")
                except Exception as e:
                    print(f"[GOOGLE FORMS] Warning: Could not share form: {e}")

            # Build response URLs
            form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
            responder_url = f"https://docs.google.com/forms/d/{form_id}/viewform"

            form_info = {
                "form_id": form_id,
                "form_url": form_url,
                "responder_url": responder_url,
                "candidate_name": exam.candidate_name,
                "candidate_email": exam.candidate_email,
                "total_questions": len(exam.questions),
            }

            print(f"[GOOGLE FORMS] ✅ Form ready! Responder URL: {responder_url}")
            return form_info

        except HttpError as e:
            print(f"[GOOGLE FORMS] API Error creating form: {e}")
            return None
        except Exception as e:
            print(f"[GOOGLE FORMS] Unexpected error creating form: {e}")
            return None

    def get_form_responses(self, form_id: str, exam: TechnicalExam | None = None) -> list[dict[str, Any]]:
        """Fetch all responses from a Google Form and optionally auto-grade them."""
        if not self.is_available:
            print("[GOOGLE FORMS] Not available. Cannot fetch responses.")
            return []

        try:
            response = self.forms_service.forms().responses().list(formId=form_id).execute()
            raw_responses = response.get("responses", [])

            if not raw_responses:
                print(f"[GOOGLE FORMS] No responses found for form '{form_id}'.")
                return []

            print(f"[GOOGLE FORMS] Found {len(raw_responses)} response(s) for form '{form_id}'.")

            graded_responses = []
            for resp in raw_responses:
                response_id = resp.get("responseId", "unknown")
                create_time = resp.get("createTime", "unknown")
                answers = resp.get("answers", {})

                total_score = 0
                total_questions = 0
                answer_details = []

                for question_id, answer_data in answers.items():
                    total_questions += 1
                    text_answers = answer_data.get("textAnswers", {}).get("answers", [])
                    chosen_answer = text_answers[0].get("value", "") if text_answers else ""
                    grade = answer_data.get("grade", {})
                    score = grade.get("score", 0)
                    correct = grade.get("correct", False)
                    total_score += score

                    answer_details.append({
                        "question_id": question_id,
                        "chosen_answer": chosen_answer,
                        "score": score,
                        "correct": correct,
                    })

                max_score = total_questions  # each question worth 1 point
                percentage = round((total_score / max_score) * 100) if max_score > 0 else 0

                graded_responses.append({
                    "response_id": response_id,
                    "submitted_at": create_time,
                    "answers": answer_details,
                    "total_score": total_score,
                    "max_score": max_score,
                    "percentage": percentage,
                    "passed": percentage >= settings.PASS_THRESHOLD,
                })

            return graded_responses

        except HttpError as e:
            print(f"[GOOGLE FORMS] API Error fetching responses: {e}")
            return []
        except Exception as e:
            print(f"[GOOGLE FORMS] Unexpected error fetching responses: {e}")
            return []


def create_forms_for_candidates(exams: list[TechnicalExam]) -> list[dict]:
    """Batch create Google Forms for multiple candidates.

    Returns list of form_info dicts for all successfully created forms.
    """
    manager = GoogleFormsManager()
    if not manager.is_available:
        print("[GOOGLE FORMS] Google Forms not configured. Skipping batch form creation.")
        return []

    created_forms = []
    for exam in exams:
        form_info = manager.create_exam_form(exam)
        if form_info:
            created_forms.append(form_info)

    print(f"[GOOGLE FORMS] Batch complete: {len(created_forms)}/{len(exams)} forms created successfully.")
    return created_forms


def fetch_form_responses(form_id: str) -> list[dict]:
    """Convenience function to fetch and grade responses for a single form."""
    manager = GoogleFormsManager()
    if not manager.is_available:
        return []
    return manager.get_form_responses(form_id)


if __name__ == "__main__":
    print("[GOOGLE FORMS] Module loaded. Testing availability...")
    manager = GoogleFormsManager()
    if manager.is_available:
        print("[GOOGLE FORMS] ✅ Google Forms API is ready!")
    else:
        print("[GOOGLE FORMS] ⚠️ Google Forms API not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE in .env")
    print("[GOOGLE FORMS] Module test complete.")
