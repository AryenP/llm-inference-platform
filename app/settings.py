from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces cleared so the `model` field doesn't collide with pydantic's model_* API
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    model: str = "/workspace/models/qwen3-8b"
    model_awq: str = "/workspace/models/qwen3-8b-awq"
    embed_model: str = "/workspace/models/bge-m3"
    rerank_model: str = "/workspace/models/bge-reranker"
    vllm_url: str = "http://localhost:8000/v1"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    request_timeout_s: float = 300.0


settings = Settings()
