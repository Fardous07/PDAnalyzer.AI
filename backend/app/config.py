import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Political Discourse Analysis"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_RELOAD: bool = Field(default=True)

    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=9018)
    DB_NAME: str = Field(default="pda")
    DB_USERNAME: str = Field(default="postgres")
    DB_PASSWORD: Optional[str] = Field(default=None)
    DATABASE_URL: Optional[str] = Field(default=None)

    DB_POOL_SIZE: int = Field(default=5)
    DB_MAX_OVERFLOW: int = Field(default=10)
    DB_POOL_TIMEOUT: int = Field(default=30)
    DB_POOL_RECYCLE: int = Field(default=3600)
    DB_POOL_PRE_PING: bool = Field(default=True)
    DB_ECHO: bool = Field(default=False)

    SECRET_KEY: Optional[str] = Field(default=None)
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)

    DEFAULT_MAX_SPEECHES: int = Field(default=50)
    DEFAULT_MAX_FILE_SIZE: int = Field(default=100_000_000)

    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    GROQ_API_KEY: Optional[str] = Field(default=None)
    DEFAULT_LLM_PROVIDER: str = Field(default="openai")
    DEFAULT_LLM_MODEL: str = Field(default="gpt-4o-mini")

    EMBEDDING_BACKEND: str = Field(default="openai")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    LOCAL_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    WHISPER_MODEL: str = Field(default="base")

    UPLOAD_FOLDER: str = Field(default="./uploads")
    CHROMA_DB_PATH: str = Field(default="./chroma_db")

    CORS_ORIGINS: str = Field(default="http://localhost:5173")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])

    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: Optional[str] = Field(default=None)
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    RATE_LIMIT_ENABLED: bool = Field(default=False)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)

    SECURE_COOKIES: bool = Field(default=False)
    SAME_SITE_COOKIES: str = Field(default="lax")

    FRONTEND_URL: str = Field(default="http://localhost:5173")
    VITE_API_BASE_URL: str = Field(default="http://localhost:8000")

    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(default=0.45)
    MAX_SENTENCES_PER_SEGMENT: int = Field(default=4)
    KEY_STATEMENT_CONFIDENCE_MIN: float = Field(default=0.60)
    KEY_STATEMENT_SIGNAL_MIN: float = Field(default=50.0)
    KEY_STATEMENT_CODES_MIN: int = Field(default=2)

    ENABLE_RESEARCH_ANALYSIS: bool = Field(default=False)
    RESEARCH_LLM_PROVIDER: str = Field(default="openai")
    RESEARCH_LLM_MODEL: str = Field(default="gpt-4o-mini")
    RESEARCH_MAX_STATEMENTS_PER_CALL: int = Field(default=10)
    RESEARCH_TIMEOUT_SECONDS: int = Field(default=120)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def construct_database_url(cls, v: Any, info: Any) -> Any:
        values = info.data

        if v:
            vv = str(v)
            if "${" in vv and isinstance(values, dict):
                for key, val in values.items():
                    if isinstance(val, (str, int)):
                        vv = vv.replace(f"${{{key}}}", str(val))
            return vv

        username = str(values.get("DB_USERNAME") or "postgres")
        password = values.get("DB_PASSWORD")
        host = str(values.get("DB_HOST") or "localhost")
        port = int(values.get("DB_PORT") or 9018)
        database = str(values.get("DB_NAME") or "pda")

        if password:
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        return f"postgresql://{username}@{host}:{port}/{database}"

    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def require_secret_in_prod(cls, v: Any, info: Any) -> Any:
        env = str(info.data.get("ENVIRONMENT", "development")).lower()
        if env in ("production", "prod") and not v:
            raise ValueError("SECRET_KEY must be set in production")
        return v

    @field_validator("DEFAULT_LLM_MODEL", mode="before")
    @classmethod
    def validate_llm_model(cls, v: Any) -> str:
        model = str(v or "gpt-4o-mini").strip()
        if model == "gpt-5.2":
            return "gpt-4o-mini"
        return model

    @property
    def cors_origins_list(self) -> List[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        if not raw:
            return ["http://localhost:5173"]

        if raw.startswith("["):
            try:
                origins = json.loads(raw)
                if isinstance(origins, list):
                    return [str(x) for x in origins if str(x).strip()]
            except Exception:
                pass

        return [x.strip() for x in raw.split(",") if x.strip()]

    def is_production(self) -> bool:
        return str(self.ENVIRONMENT).lower() in ("production", "prod")

    def is_development(self) -> bool:
        return str(self.ENVIRONMENT).lower() in ("development", "dev")

    def is_testing(self) -> bool:
        return str(self.ENVIRONMENT).lower() in ("testing", "test")


settings = Settings()


def ensure_runtime_directories() -> None:
    Path(settings.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)


__all__ = ["settings", "Settings", "ensure_runtime_directories"]