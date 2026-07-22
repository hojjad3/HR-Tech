import os
import re
from pathlib import Path
import pdfplumber
from pydantic import BaseModel


class ParsedResume(BaseModel):
    file_name: str
    candidate_name: str
    candidate_email: str
    raw_text: str
    char_count: int


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def extract_email(text: str) -> str:
    matches = EMAIL_REGEX.findall(text)
    if matches:
        return matches[0]
    return "unknown@candidate.com"


def extract_candidate_name(text: str, file_path: str) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) <= 4 and not any(char.isdigit() for char in first_line):
            return first_line
    stem = Path(file_path).stem
    clean_stem = stem.replace("_", " ").replace("-", " ").title()
    return clean_stem


def parse_resume_pdf(file_path: str) -> ParsedResume:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Resume PDF file not found: {file_path}")

    print(f"[PARSER] Extracting text from resume PDF: {os.path.basename(file_path)}")
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text.append(text)

    extracted_text = "\n\n".join(full_text).strip()
    email = extract_email(extracted_text)
    name = extract_candidate_name(extracted_text, file_path)

    parsed = ParsedResume(
        file_name=os.path.basename(file_path),
        candidate_name=name,
        candidate_email=email,
        raw_text=extracted_text,
        char_count=len(extracted_text),
    )

    print(f"[PARSER] Successfully parsed '{parsed.file_name}': Candidate='{parsed.candidate_name}', Email='{parsed.candidate_email}', Chars={parsed.char_count}")
    return parsed


if __name__ == "__main__":
    test_text = """
    Jane Doe
    Software Systems Engineer
    Email: jane.doe@example.com | Phone: +1-555-0199
    
    Summary:
    Experienced Backend Developer with 5 years of Python, FastEmbed, RAG, and PostgreSQL experience.
    """
    email = extract_email(test_text)
    name = extract_candidate_name(test_text, "jane_doe_resume.pdf")
    print(f"[PARSER TEST] Extracted Name: {name}")
    print(f"[PARSER TEST] Extracted Email: {email}")
    assert email == "jane.doe@example.com"
    assert name == "Jane Doe"
    print("[PARSER TEST] All parser regex tests passed successfully!")
