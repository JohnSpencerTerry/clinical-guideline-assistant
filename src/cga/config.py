from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "clinical-guideline-assistant"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    vector_store_dir: str = "./data/processed/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
