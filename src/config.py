import os
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    def print_status(self) -> None:
        print("[CONFIG] Environment Settings Loaded:")
        print(f"  - LLM Provider: {self.LLM_PROVIDER}")
        print(f"  - LLM Model: {self.LLM_MODEL}")
        print(f"  - Embedding Model: {self.EMBEDDING_MODEL}")
        print(f"  - Chroma Directory: {self.CHROMA_PERSIST_DIR}")
        print(f"  - OpenAI API Key: {'Configured' if self.OPENAI_API_KEY else 'Not Set'}")
        print(f"  - Groq API Key: {'Configured' if self.GROQ_API_KEY else 'Not Set'}")
        print(f"  - Resend API Key: {'Configured' if self.RESEND_API_KEY else 'Not Set'}")


settings = Settings()

if __name__ == "__main__":
    settings.print_status()
