from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", extra="ignore")

    openai_api_key: str
    openai_embed_model: str
    openai_model: str
    openai_fast_model: str
    openai_guard_model: str
    judge_model: str = ""  # evals: DeepEval judge; "groq/<model>" uses Groq; empty = openai_fast_model
    groq_api_key: str = ""
    eval_api: str = "http://127.0.0.1:8000"
    web_search_enabled: bool = False  # per-run opt-in still required (RunIn.web_search)
    tavily_api_key: str = ""
    web_search_timeout_seconds: float = 20
    qdrant_cluster_endpoint: str
    qdrant_api_key: str
    qdrant_collection: str
    hybrid_search: bool = False  # dense + bm25 keyword search fused by RRF; needs a migrated collection (ingest.py --migrate-to)
    reranker_model: str = ""  # "" = off; "jina:<model>" = hosted Jina; otherwise a local Hugging Face cross-encoder id
    jena_ai_api_key: str = ""  # env var is spelled Jena_AI_API_KEY (Jina AI); matching is case-insensitive
    database_url: str
    jwt_secret: str
    access_token_minutes: int = 60
