
"""
CONFIGURATION MODULE (Pydantic v2)
LOCATION: backend/app/config.py
"""

import json
from typing import Optional, List
from pathlib import Path

from pydantic import Field, field_validator, ConfigDict  # v2 API
from pydantic_settings import BaseSettings, SettingsConfigDict  # v2 API


class Settings(BaseSettings):
    # ── APP ──────────────────────────────────────────────────────────────────────
    APP_NAME: str = "Political Discourse Analysis"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")  # development|production|testing
    DEBUG: bool = Field(default=False, env="DEBUG")
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_RELOAD: bool = Field(default=True, env="API_RELOAD")  # set False in prod

    # ── DATABASE ────────────────────────────────────────────────────────────────
    # Prefer DATABASE_URL. Fallback builds it from parts if not provided.
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=9018, env="DB_PORT")
    DB_NAME: str = Field(default="pda", env="DB_NAME")
    DB_USERNAME: str = Field(default="postgres", env="DB_USERNAME")
    DB_PASSWORD: Optional[str] = Field(default=None, env="DB_PASSWORD")  # NO DEFAULT SECRET
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")

    DB_POOL_SIZE: int = Field(default=5, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=10, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    DB_POOL_RECYCLE: int = Field(default=3600, env="DB_POOL_RECYCLE")
    DB_POOL_PRE_PING: bool = Field(default=True, env="DB_POOL_PRE_PING")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")

    # ── AUTH / JWT ──────────────────────────────────────────────────────────────
    SECRET_KEY: Optional[str] = Field(default=None, env="SECRET_KEY")  # MUST be set in prod
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    # ── USER LIMITS ─────────────────────────────────────────────────────────────
    DEFAULT_MAX_SPEECHES: int = Field(default=50, env="DEFAULT_MAX_SPEECHES")
    DEFAULT_MAX_FILE_SIZE: int = Field(default=100_000_000, env="DEFAULT_MAX_FILE_SIZE")

    # ── LLM / PROVIDERS ─────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    DEFAULT_LLM_MODEL: str = Field(default="gpt-4o-mini", env="DEFAULT_LLM_MODEL")

    # ── EMBEDDINGS ──────────────────────────────────────────────────────────────
    EMBEDDING_BACKEND: str = Field(default="openai", env="EMBEDDING_BACKEND")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    LOCAL_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="LOCAL_EMBEDDING_MODEL")

    # ── MODELS ─────────────────────────────────────────────────────────────────
    WHISPER_MODEL: str = Field(default="base", env="WHISPER_MODEL")

    # ── STORAGE PATHS (no side-effects here) ───────────────────────────────────
    UPLOAD_FOLDER: str = Field(default="./uploads", env="UPLOAD_FOLDER")
    CHROMA_DB_PATH: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = Field(default="http://localhost:5173", env="CORS_ORIGINS")  # CSV or JSON array
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], env="CORS_ALLOW_METHODS")
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")

    # ── LOGGING ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")

    # ── RATE LIMITING ──────────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    # ── SECURITY ───────────────────────────────────────────────────────────────
    SECURE_COOKIES: bool = Field(default=False, env="SECURE_COOKIES")
    SAME_SITE_COOKIES: str = Field(default="lax", env="SAME_SITE_COOKIES")  # lax|strict|none

    # ── FRONTEND ───────────────────────────────────────────────────────────────
    FRONTEND_URL: str = Field(default="http://localhost:5173", env="FRONTEND_URL")
    VITE_API_BASE_URL: str = Field(default="http://localhost:8000", env="VITE_API_BASE_URL")

    # ── ANALYSIS SETTINGS ──────────────────────────────────────────────────────
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(default=0.45, env="SEMANTIC_SIMILARITY_THRESHOLD")
    MAX_SENTENCES_PER_SEGMENT: int = Field(default=4, env="MAX_SENTENCES_PER_SEGMENT")
    KEY_STATEMENT_CONFIDENCE_MIN: float = Field(default=0.70, env="KEY_STATEMENT_CONFIDENCE_MIN")
    KEY_STATEMENT_SIGNAL_MIN: float = Field(default=65.0, env="KEY_STATEMENT_SIGNAL_MIN")
    KEY_STATEMENT_CODES_MIN: int = Field(default=2, env="KEY_STATEMENT_CODES_MIN")

    # ── Pydantic v2 settings config (replaces v1's class Config) ───────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Validators (v2) ────────────────────────────────────────────────────────
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def construct_database_url(cls, v, info):
        # If DATABASE_URL provided, allow variable expansion like ${DB_HOST}
        values = info.data
        if v:
            if isinstance(v, str) and "${" in v:
                for key, val in values.items():
                    if isinstance(val, (str, int)):
                        v = v.replace(f"${{{key}}}", str(val))
            return v

        # Build from parts (useful in dev); in prod you should set DATABASE_URL
        username = values.get("DB_USERNAME", "postgres")
        password = values.get("DB_PASSWORD")
        host = values.get("DB_HOST", "localhost")
        port = values.get("DB_PORT", 9018)
        database = values.get("DB_NAME", "pda")

        if password:
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        return f"postgresql://{username}@{host}:{port}/{database}"

    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def require_secret_in_prod(cls, v, info):
        env = str(info.data.get("ENVIRONMENT", "development")).lower()
        if env in ("production", "prod") and not v:
            raise ValueError("SECRET_KEY must be set in production")
        return v

    # ── Helpers ────────────────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS:
            return ["http://localhost:5173"]
        # JSON list support
        try:
            if self.CORS_ORIGINS.strip().startswith("["):
                origins = json.loads(self.CORS_ORIGINS)
                if isinstance(origins, list):
                    return [str(origin) for origin in origins]
        except Exception:
            pass
        # CSV
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ("production", "prod")

    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() in ("development", "dev")

    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() in ("testing", "test")


settings = Settings()


def ensure_runtime_directories() -> None:
    """
    Create directories that must exist at runtime.
    Call this from app startup (not at import time).
    """
    Path(settings.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)


__all__ = ["settings", "Settings", "ensure_runtime_directories"]
