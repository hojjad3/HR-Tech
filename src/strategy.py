import json
import sys
from pydantic import BaseModel, Field
from openai import OpenAI
from src.config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class HiringStrategy(BaseModel):
    product_summary: str = Field(description="Brief technical description of what is being built")
    job_title: str = Field(description="The ideal target job role")
    must_have_skills: list[str] = Field(description="Essential technologies/frameworks needed")
    nice_to_have_skills: list[str] = Field(default_factory=list, description="Secondary or advanced skills")
    evaluation_criteria: list[str] = Field(description="Specific evaluation benchmarks directly tied to constructing this product")


STRATEGY_SYSTEM_PROMPT = """You are an expert AI Product Manager & Technical Hiring Architect.
The user will describe an AI product, feature, or business idea in plain or informal language (in Arabic or English).

Your task:
1. Analyze the AI product concept and understand its core technical architecture.
2. Reverse-engineer the EXACT technical skills, tools, frameworks, and experience required to build this specific product from scratch.
3. Output a structured hiring strategy in JSON format with:
   - "product_summary": Brief technical summary of what is being built.
   - "job_title": The ideal target job role needed to build this product.
   - "must_have_skills": Essential technologies/frameworks needed.
   - "nice_to_have_skills": Secondary or advanced skills that add value.
   - "evaluation_criteria": Specific evaluation benchmarks directly tied to constructing this product.

You MUST respond strictly with a JSON object matching this schema:
{
  "product_summary": "Brief technical description...",
  "job_title": "Ideal Job Role",
  "must_have_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill1", "skill2"],
  "evaluation_criteria": ["criterion1", "criterion2"]
}
"""


def _get_llm_client() -> tuple[OpenAI | None, str]:
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        print("[STRATEGY] Using Groq API client...")
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        ), settings.LLM_MODEL
    elif settings.OPENAI_API_KEY:
        print("[STRATEGY] Using OpenAI API client...")
        return OpenAI(api_key=settings.OPENAI_API_KEY), "gpt-4o-mini"
    else:
        print("[STRATEGY] No LLM API key detected. Using fallback mock client for testing.")
        return None, "mock"


def generate_hiring_strategy(user_prompt: str) -> HiringStrategy:
    print(f"[STRATEGY] Reverse-engineering Product Concept from prompt:\n  '{user_prompt}'")
    client, model = _get_llm_client()

    if client is None:
        mock_strategy = HiringStrategy(
            product_summary="نظام مساعد قانوني ذكي (Legal AI Assistant) يعمل بتقنية RAG على النصوص القانونية العربية والأحكام القضائية.",
            job_title="AI/RAG System Engineer (Arabic NLP Specialist)",
            must_have_skills=[
                "Python",
                "Retrieval-Augmented Generation (RAG)",
                "Vector Databases (ChromaDB / Qdrant)",
                "Arabic Natural Language Processing (NLP)",
                "LLM Integration & Fine-Tuning",
            ],
            nice_to_have_skills=[
                "Hybrid Search (BM25 + Dense Embeddings)",
                "Cross-Encoder Re-ranking",
                "FastAPI / Backend Architecture",
            ],
            evaluation_criteria=[
                "خبرة عملية مثبتة في بناء أنظمة RAG للغة العربية معالجة للمستندات الطويلة",
                "فهم آليات الـ Chunking والـ Semantic Search للتشريعات والنصوص القانونية",
                "القدرة على التعامل مع أطر عمل الـ Text Preprocessing والاستخراج من ملفات PDF",
            ],
        )
        print("[STRATEGY] Generated Fallback Product-to-Hiring Strategy successfully.")
        return mock_strategy

    prompt = f"Product Concept / Business Request:\n\"\"\"{user_prompt}\"\"\""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    strategy = HiringStrategy(**data)

    print(f"[STRATEGY] Reverse-engineered Product Strategy for Role: '{strategy.job_title}'")
    print(f"  - Product Summary: {strategy.product_summary}")
    print(f"  - Must-Have Skills: {', '.join(strategy.must_have_skills)}")
    print(f"  - Evaluation Criteria: {len(strategy.evaluation_criteria)} benchmarks")
    return strategy


if __name__ == "__main__":
    arabic_prompt = "يا سيدي أنا بدي أبني تطبيق قانوني بيساعد المحامين بالأردن، بحيث يرفعوا القضية والـ AI يعمل تحليلات للنسخ والأحكام القديمة ويرتبلهم ملخص واستشارات بناءً على القانون المدني."
    strat = generate_hiring_strategy(arabic_prompt)
    print("\n--- Reverse-Engineered Hiring Strategy JSON ---")
    print(json.dumps(strat.model_dump(), indent=2, ensure_ascii=False))
