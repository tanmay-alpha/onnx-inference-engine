"""Centralized configuration for the Crucible server.

Every setting here is read by server code; nothing is declared "for later".
Values come from the process environment (and an optional ``.env`` file in the
working directory).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The development JWT secret. Refused when ENVIRONMENT=production.
DEV_SECRET_KEY = "dev-secret-key-change-in-production-please-use-a-real-secret"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # -- Application ----------------------------------------------------------
    ENVIRONMENT: str = Field(default="development", description="development, staging, production or test")
    APP_NAME: str = Field(default="Crucible Inference Server")
    APP_VERSION: str = Field(default="1.0.0")

    # -- Database -------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./crucible.db",
        description=(
            "SQLAlchemy URL. sqlite+aiosqlite:///... or postgresql://... "
            "(asyncpg is used for async sessions, psycopg2 for sync CRUD)."
        ),
    )
    DB_STATEMENT_CACHE_SIZE: int = Field(
        default=0, description="asyncpg prepared statement cache size (0 for PgBouncer/Supabase pooler)"
    )

    # -- Authentication -------------------------------------------------------
    CRUCIBLE_SECRET_KEY: str = Field(default=DEV_SECRET_KEY, description="JWT signing secret")
    CRUCIBLE_TOKEN_EXPIRE_MINUTES: int = Field(default=60)
    CRUCIBLE_API_KEY: str = Field(
        default="",
        description="Optional operator key (X-API-Key). Empty disables operator access; "
                    "users still authenticate with JWTs or their own API keys.",
    )
    CRUCIBLE_API_KEY_PREFIX: str = Field(default="cr_", description="Prefix of generated user API keys")

    # -- Storage, limits, inference ------------------------------------------
    CRUCIBLE_MODEL_DIR: str = Field(default="/tmp/models", description="Directory for uploaded ONNX files")
    # Peak parse memory is ~2x the file (protobuf bytes + parsed model), so
    # 50 MB keeps one analysis well inside a 512 MB instance.
    MAX_UPLOAD_BYTES: int = Field(default=50 * 1024 * 1024, description="Max .onnx upload size (/convert, /validate)")
    MAX_CONCURRENT_ANALYSES: int = Field(
        default=1, description="Model analyses (/convert, /validate) in flight at once; 503 + Retry-After beyond"
    )
    MAX_GRAPH_NODES: int = Field(default=2000, description="Max nodes in an uploaded model's graph")
    MAX_MODELS_PER_USER: int = Field(default=20, description="Stored models per user (operator key exempt)")
    MAX_STORAGE_BYTES_PER_USER: int = Field(
        default=200 * 1024 * 1024, description="Total stored model bytes per user (operator key exempt)"
    )
    MAX_REQUEST_BODY_BYTES: int = Field(default=16 * 1024 * 1024, description="Max body size for every non-upload route")
    MAX_INPUT_ELEMENTS: int = Field(default=4_000_000, description="Max elements in one /infer input tensor")
    MAX_TENSOR_ELEMENTS: int = Field(
        default=16_000_000,
        description="Max elements of any tensor the engine allocates during a run (untrusted-model guard; 64 MB of float32)",
    )
    MAX_TOTAL_ELEMENTS: int = Field(
        default=64_000_000,
        description="Max elements across all live intermediate tensors of one run (256 MB of float32)",
    )
    MAX_OUTPUT_ELEMENTS: int = Field(default=1_000_000, description="Max output elements returned by /infer (and per batch)")
    MAX_BATCH_SIZE: int = Field(default=32, description="Max requests in one /inference/batch call")
    INFERENCE_TIMEOUT_SEC: float = Field(
        default=60.0, gt=0, description="Deadline of one native run; the engine stops at it and the request gets 504"
    )
    MAX_CONCURRENT_INFERENCES: int = Field(default=2, description="Native runs allowed in flight at once")
    MODEL_CACHE_SIZE: int = Field(default=4, ge=0, description="Loaded native models kept in memory (LRU; 0 disables)")

    # -- Abuse protection (per process; not shared across instances) ----------
    TRUST_PROXY_HEADERS: bool = Field(
        default=False,
        description="Take the client IP from the first X-Forwarded-For entry. Enable only behind a proxy "
                    "that sets that entry itself (Render does); otherwise clients can spoof it",
    )
    AUTH_RATE_LIMIT_PER_IP: int = Field(default=30, description="/auth/login + /auth/register attempts per IP per window")
    AUTH_RATE_LIMIT_PER_IP_EMAIL: int = Field(default=10, description="/auth/login attempts per IP+email per window")
    AUTH_RATE_WINDOW_SEC: float = Field(default=300.0, gt=0, description="Sliding window of the auth limits")

    # -- CORS -----------------------------------------------------------------
    CRUCIBLE_CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins",
    )

    # -- Logging --------------------------------------------------------------
    CRUCIBLE_LOG_LEVEL: str = Field(default="INFO")
    CRUCIBLE_LOG_FORMAT: str = Field(default="console", description="console or json")

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CRUCIBLE_CORS_ORIGINS.split(",") if o.strip()]

    @field_validator("ENVIRONMENT")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        valid = {"development", "staging", "production", "test"}
        if v.lower() not in valid:
            raise ValueError(f"ENVIRONMENT must be one of {sorted(valid)}")
        return v.lower()

    @field_validator(
        "MAX_CONCURRENT_INFERENCES", "MAX_BATCH_SIZE", "MAX_INPUT_ELEMENTS", "MAX_TENSOR_ELEMENTS",
        "MAX_TOTAL_ELEMENTS", "MAX_OUTPUT_ELEMENTS", "MAX_CONCURRENT_ANALYSES", "MAX_GRAPH_NODES",
        "MAX_MODELS_PER_USER", "MAX_STORAGE_BYTES_PER_USER", "AUTH_RATE_LIMIT_PER_IP",
        "AUTH_RATE_LIMIT_PER_IP_EMAIL",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    def validate_production_security(self) -> None:
        """Refuse to start in production with settings the server actually relies on left at dev values."""
        if self.ENVIRONMENT != "production":
            return
        if self.CRUCIBLE_SECRET_KEY == DEV_SECRET_KEY or len(self.CRUCIBLE_SECRET_KEY) < 32:
            raise ValueError(
                "CRUCIBLE_SECRET_KEY must be set to a random string of at least 32 characters in production."
            )
        if "DATABASE_URL" not in self.model_fields_set:
            raise ValueError("DATABASE_URL must be set explicitly in production.")
        if self.CRUCIBLE_API_KEY and len(self.CRUCIBLE_API_KEY) < 16:
            raise ValueError("CRUCIBLE_API_KEY, when set, must be at least 16 characters in production.")


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_production_security()
    return settings
