from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import OpenAI


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    RESEND_API_KEY: str | None = None
    SENDER_EMAIL: str = "onboarding@resend.dev"
    LLM_PROVIDER: str = "groq"  # "groq" or "openai"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    PASS_THRESHOLD: int = 75

    def print_status(self) -> None:
        print("[CONFIG] Environment Settings Loaded:")
        print(f"  - LLM Provider: {self.LLM_PROVIDER}")
        print(f"  - LLM Model: {self.LLM_MODEL}")
        print(f"  - Embedding Model: {self.EMBEDDING_MODEL}")
        print(f"  - Chroma Directory: {self.CHROMA_PERSIST_DIR}")
        print(f"  - Pass Threshold: {self.PASS_THRESHOLD}")
        print(f"  - OpenAI API Key: {'Configured' if self.OPENAI_API_KEY else 'Not Set'}")
        print(f"  - Groq API Key: {'Configured' if self.GROQ_API_KEY else 'Not Set'}")
        print(f"  - Resend API Key: {'Configured' if self.RESEND_API_KEY else 'Not Set'}")


settings = Settings()


def get_llm_client() -> tuple[OpenAI | None, str]:
    """Shared LLM client factory — single source of truth for all modules."""
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        print("[LLM] Using Groq API client...")
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        ), settings.LLM_MODEL
    elif settings.OPENAI_API_KEY:
        print("[LLM] Using OpenAI API client...")
        return OpenAI(api_key=settings.OPENAI_API_KEY), "gpt-4o-mini"
    else:
        print("[LLM] No LLM API key detected. Using fallback mock mode.")
        return None, "mock"


if __name__ == "__main__":
    settings.print_status()
    client, model = get_llm_client()
    print(f"  - LLM Client: {type(client).__name__ if client else 'Mock'}")
    print(f"  - Active Model: {model}")
