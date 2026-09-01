from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces cleared so the `model` field doesn't collide with pydantic's model_* API
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # No default. The right checkpoint differs per host — AWQ on a 12 GB card, bf16
    # on the rented one — and a default that silently resolves to a missing path is
    # worse than refusing to start.
    model: str
    model_awq: str | None = None
    embed_model: str | None = None
    rerank_model: str | None = None

    vllm_url: str = "http://localhost:8000/v1"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    request_timeout_s: float = 300.0


settings = Settings()
