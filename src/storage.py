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
