import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

RESULTS_DIR = "./results"
DB_PATH = os.path.join(RESULTS_DIR, "history.db")


def init_db() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            timestamp TEXT,
            prompt TEXT,
            job_title TEXT,
            product_summary TEXT,
            strategy_json TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            candidate_name TEXT,
            candidate_email TEXT,
            match_score INTEGER,
            passed INTEGER,
            reasoning TEXT,
            exam_json TEXT,
            email_sent INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS google_forms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            candidate_name TEXT,
            candidate_email TEXT,
            form_id TEXT UNIQUE,
            form_url TEXT,
            responder_url TEXT,
            total_questions INTEGER,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS form_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_id TEXT,
            response_id TEXT UNIQUE,
            submitted_at TEXT,
            total_score INTEGER,
            max_score INTEGER,
            percentage INTEGER,
            passed INTEGER,
            answers_json TEXT,
            fetched_at TEXT,
            FOREIGN KEY (form_id) REFERENCES google_forms(form_id)
        )
        """)

        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[STORAGE] Database initialization error: {e}")


def save_screening_session(
    prompt: str,
    strategy_dict: dict,
    candidate_records: list[dict],
) -> str:
    init_db()
    session_id = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    job_title = strategy_dict.get("job_title", "Unknown Role")
    product_summary = strategy_dict.get("product_summary", "")
    strategy_json = json.dumps(strategy_dict, ensure_ascii=False)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, timestamp, prompt, job_title, product_summary, strategy_json),
        )

        for rec in candidate_records:
            exam_json = json.dumps(rec.get("exam"), ensure_ascii=False) if rec.get("exam") else None
            cursor.execute(
                """
                INSERT INTO candidates (session_id, candidate_name, candidate_email, match_score, passed, reasoning, exam_json, email_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    rec.get("candidate_name", "Unknown"),
                    rec.get("candidate_email", ""),
                    int(rec.get("match_score", 0)),
                    1 if rec.get("passed") else 0,
                    rec.get("reasoning", ""),
                    exam_json,
                    1 if rec.get("email_dispatched") else 0,
                ),
            )

        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[STORAGE] Error saving session to database: {e}")

    # Save JSON snapshot file in ./results/
    try:
        json_filepath = os.path.join(RESULTS_DIR, f"{session_id}.json")
        snapshot_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "strategy": strategy_dict,
            "candidates": candidate_records,
        }
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[STORAGE] Error saving JSON snapshot: {e}")

    print(f"[STORAGE] Successfully saved screening session '{session_id}' to SQLite DB and JSON snapshot.")
    return session_id


def save_google_form(
    session_id: str,
    form_info: dict,
) -> None:
    """Save Google Form metadata to the database."""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO google_forms
            (session_id, candidate_name, candidate_email, form_id, form_url, responder_url, total_questions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                form_info.get("candidate_name", ""),
                form_info.get("candidate_email", ""),
                form_info["form_id"],
                form_info.get("form_url", ""),
                form_info.get("responder_url", ""),
                form_info.get("total_questions", 0),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        conn.close()
        print(f"[STORAGE] Saved Google Form '{form_info['form_id']}' for candidate '{form_info.get('candidate_name')}'")
    except sqlite3.Error as e:
        print(f"[STORAGE] Error saving Google Form: {e}")


def save_form_responses(form_id: str, responses: list[dict]) -> None:
    """Save fetched Google Form responses to the database."""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for resp in responses:
            cursor.execute(
                """
                INSERT OR REPLACE INTO form_responses
                (form_id, response_id, submitted_at, total_score, max_score, percentage, passed, answers_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    form_id,
                    resp.get("response_id", ""),
                    resp.get("submitted_at", ""),
                    resp.get("total_score", 0),
                    resp.get("max_score", 0),
                    resp.get("percentage", 0),
                    1 if resp.get("passed") else 0,
                    json.dumps(resp.get("answers", []), ensure_ascii=False),
                    fetched_at,
                ),
            )

        conn.commit()
        conn.close()
        print(f"[STORAGE] Saved {len(responses)} response(s) for form '{form_id}'")
    except sqlite3.Error as e:
        print(f"[STORAGE] Error saving form responses: {e}")


def get_forms_for_session(session_id: str) -> list[dict]:
    """Get all Google Forms created for a specific session."""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM google_forms WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        forms = []
        for r in rows:
            forms.append({
                "candidate_name": r["candidate_name"],
                "candidate_email": r["candidate_email"],
                "form_id": r["form_id"],
                "form_url": r["form_url"],
                "responder_url": r["responder_url"],
                "total_questions": r["total_questions"],
                "created_at": r["created_at"],
            })
        return forms
    except sqlite3.Error as e:
        print(f"[STORAGE] Error fetching forms: {e}")
        return []


def get_all_forms() -> list[dict]:
    """Get all Google Forms across all sessions."""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT gf.*, s.job_title 
            FROM google_forms gf 
            LEFT JOIN sessions s ON gf.session_id = s.session_id 
            ORDER BY gf.created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        forms = []
        for r in rows:
            # Count responses
            resp_count = get_response_count(r["form_id"])
            forms.append({
                "Session ID": r["session_id"],
                "Candidate": r["candidate_name"],
                "Email": r["candidate_email"],
                "Form URL": r["responder_url"],
                "Questions": r["total_questions"],
                "Responses": resp_count,
                "Created": r["created_at"],
                "Job Title": r["job_title"] or "N/A",
                "form_id": r["form_id"],
            })
        return forms
    except sqlite3.Error as e:
        print(f"[STORAGE] Error fetching all forms: {e}")
        return []


def get_response_count(form_id: str) -> int:
    """Get the number of stored responses for a form."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM form_responses WHERE form_id = ?", (form_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except sqlite3.Error:
        return 0


def get_form_responses_from_db(form_id: str) -> list[dict]:
    """Get stored form responses from the database."""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM form_responses WHERE form_id = ? ORDER BY submitted_at DESC",
            (form_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        responses = []
        for r in rows:
            responses.append({
                "response_id": r["response_id"],
                "submitted_at": r["submitted_at"],
                "total_score": r["total_score"],
                "max_score": r["max_score"],
                "percentage": r["percentage"],
                "passed": bool(r["passed"]),
                "answers": json.loads(r["answers_json"]) if r["answers_json"] else [],
                "fetched_at": r["fetched_at"],
            })
        return responses
    except sqlite3.Error as e:
        print(f"[STORAGE] Error fetching form responses: {e}")
        return []


def get_all_sessions() -> list[dict]:
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT session_id, timestamp, job_title, product_summary FROM sessions ORDER BY timestamp DESC")
        session_rows = cursor.fetchall()

        sessions = []
        for r in session_rows:
            session_id = r["session_id"]
            cursor.execute("SELECT candidate_name, match_score, passed FROM candidates WHERE session_id = ?", (session_id,))
            cand_rows = cursor.fetchall()

            cand_names = []
            cand_statuses = []

            for c in cand_rows:
                status_icon = "PASSED ✅" if c["passed"] else "FAILED ❌"
                cand_name = c["candidate_name"] or "Unknown"
                cand_names.append(cand_name)
                cand_statuses.append(f"{cand_name}: {status_icon} ({c['match_score']}/100)")

            sessions.append({
                "Session ID": session_id,
                "Timestamp": r["timestamp"],
                "Candidates": ", ".join(cand_names) if cand_names else "N/A",
                "Screening Status": ", ".join(cand_statuses) if cand_statuses else "N/A",
                "Target Job Title": r["job_title"],
                "Product Concept": r["product_summary"],
            })

        conn.close()
        return sessions
    except sqlite3.Error as e:
        print(f"[STORAGE] Error fetching sessions: {e}")
        return []


def get_session_details(session_id: str) -> dict | None:
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            conn.close()
            return None

        cursor.execute("SELECT * FROM candidates WHERE session_id = ?", (session_id,))
        candidate_rows = cursor.fetchall()
        conn.close()

        candidates = []
        for c in candidate_rows:
            exam_dict = json.loads(c["exam_json"]) if c["exam_json"] else None
            candidates.append({
                "candidate_name": c["candidate_name"],
                "candidate_email": c["candidate_email"],
                "match_score": c["match_score"],
                "passed": bool(c["passed"]),
                "reasoning": c["reasoning"],
                "exam": exam_dict,
                "email_dispatched": bool(c["email_sent"]),
            })

        return {
            "session_id": session_row["session_id"],
            "timestamp": session_row["timestamp"],
            "prompt": session_row["prompt"],
            "strategy": json.loads(session_row["strategy_json"]),
            "candidates": candidates,
        }
    except sqlite3.Error as e:
        print(f"[STORAGE] Error fetching session details: {e}")
        return None


def export_session_to_csv(session_id: str) -> str:
    details = get_session_details(session_id)
    if not details:
        raise ValueError(f"Session '{session_id}' not found.")

    rows = []
    for c in details["candidates"]:
        rows.append({
            "Session ID": details["session_id"],
            "Timestamp": details["timestamp"],
            "Job Title": details["strategy"].get("job_title"),
            "Candidate Name": c["candidate_name"],
            "Candidate Email": c["candidate_email"],
            "Match Score": c["match_score"],
            "Passed Status": "Passed" if c["passed"] else "Failed",
            "Reasoning": c["reasoning"],
            "Has MCQ Exam": "Yes" if c["exam"] else "No",
            "Email Dispatched": "Yes" if c["email_dispatched"] else "No",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"export_{session_id}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[STORAGE] Exported session '{session_id}' to CSV: {csv_path}")
    return csv_path


def export_session_to_json(session_id: str) -> str:
    details = get_session_details(session_id)
    if not details:
        raise ValueError(f"Session '{session_id}' not found.")

    json_path = os.path.join(RESULTS_DIR, f"export_{session_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print(f"[STORAGE] Exported session '{session_id}' to JSON: {json_path}")
    return json_path


if __name__ == "__main__":
    init_db()
    print("[STORAGE TEST] Database initialized at:", DB_PATH)
    sessions = get_all_sessions()
    print(f"[STORAGE TEST] Existing session count: {len(sessions)}")
    forms = get_all_forms()
    print(f"[STORAGE TEST] Existing Google Forms count: {len(forms)}")
