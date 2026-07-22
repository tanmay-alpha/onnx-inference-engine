"""Centralized Configuration System for Crucible Server.

Parses, validates, and manages environment variables for all server modules:
  - Application & Environment settings
  - Database & Connection pooling
  - Authentication & JWT keys
  - Upload limits & Inference timeouts
  - CORS origins & Frontend URLs
  - Logging & Metrics
  - Webhooks & Retry policies
  - Machine Learning & Fraud detection thresholds
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---------------------------------------------------------------------------
    # Application & Environment
    # ---------------------------------------------------------------------------
    ENVIRONMENT: str = Field(default="development", description="App environment: development, staging, production")
    APP_NAME: str = Field(default="Crucible AI Inference Engine", description="Application display name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    HOST: str = Field(default="0.0.0.0", description="Bind host IP")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    PYTHONUNBUFFERED: int = Field(default=1, description="Python stdout buffering")
    TZ: str = Field(default="UTC", description="Server timezone")

    # ---------------------------------------------------------------------------
    # Database
    # ---------------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./crucible.db",
        description="Async SQLAlchemy database connection URL",
    )
    DB_POOL_SIZE: int = Field(default=5, description="SQLAlchemy pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, description="SQLAlchemy max pool overflow")
    DB_POOL_TIMEOUT: int = Field(default=30, description="SQLAlchemy pool checkout timeout (sec)")
    DB_STATEMENT_CACHE_SIZE: int = Field(default=0, description="Asyncpg prepared statement cache size (0 for PgBouncer)")

    # ---------------------------------------------------------------------------
    # Authentication & Security
    # ---------------------------------------------------------------------------
    CRUCIBLE_SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production-please-use-a-real-secret",
        description="JWT token signing secret key",
    )
    CRUCIBLE_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description="JWT token validity duration in minutes")
    CRUCIBLE_API_KEY: str = Field(default="crucible-development-api-key", description="Static master API key")
    CRUCIBLE_API_KEY_PREFIX: str = Field(default="cr_", description="API key string prefix")

    # ---------------------------------------------------------------------------
    # Server, Limits & Storage
    # ---------------------------------------------------------------------------
    CRUCIBLE_MODEL_DIR: str = Field(default="/tmp/models", description="Directory path for ONNX model storage")
    INFERENCE_TIMEOUT_SEC: int = Field(default=60, description="Hard timeout for model execution (sec)")
    MAX_UPLOAD_BYTES: int = Field(default=209715200, description="Max model file upload size in bytes (200MB default)")
    MAX_REQUEST_BODY_BYTES: int = Field(default=10485760, description="Max JSON request body size in bytes (10MB default)")
    MAX_INPUT_ELEMENTS: int = Field(default=50000000, description="Max input tensor element count (50M default)")

    # ---------------------------------------------------------------------------
    # Frontend & CORS
    # ---------------------------------------------------------------------------
    NEXT_PUBLIC_API_URL: str = Field(default="http://localhost:8000", description="Public API URL for web frontend")
    CRUCIBLE_CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins",
    )

    # ---------------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------------
    CRUCIBLE_LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    CRUCIBLE_LOG_FORMAT: str = Field(default="console", description="Log format: console or json")

    # ---------------------------------------------------------------------------
    # Machine Learning & Fraud Detection
    # ---------------------------------------------------------------------------
    CRUCIBLE_TRAIN_DATA: str = Field(default="data/raw/creditcard.csv", description="Path to training CSV dataset")
    CRUCIBLE_MODEL_OUTPUT: str = Field(default="models/fraud_model.onnx", description="Output path for trained ONNX model")
    FRAUD_HIGH_RISK_THRESHOLD: float = Field(default=0.7, description="High risk probability threshold")
    FRAUD_CRITICAL_RISK_THRESHOLD: float = Field(default=0.9, description="Critical risk probability threshold")

    # ---------------------------------------------------------------------------
    # Webhook Notifications
    # ---------------------------------------------------------------------------
    WEBHOOK_TIMEOUT_SEC: int = Field(default=10, description="Webhook HTTP request timeout in seconds")
    WEBHOOK_MAX_RETRIES: int = Field(default=3, description="Max delivery attempt count")
    WEBHOOK_MAX_FAILURES: int = Field(default=5, description="Consecutive failure limit before auto-disabling webhook")

    # ---------------------------------------------------------------------------
    # Docker & Infrastructure
    # ---------------------------------------------------------------------------
    POSTGRES_PASSWORD: Optional[str] = Field(default="crucible_dev_password", description="PostgreSQL dev password")
    GRAFANA_PASSWORD: Optional[str] = Field(default="admin", description="Grafana dashboard admin password")

    # Helper parsing properties
    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a trimmed list of strings."""
        if not self.CRUCIBLE_CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CRUCIBLE_CORS_ORIGINS.split(",") if origin.strip()]

    # Validators
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = {"development", "staging", "production", "test"}
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()

    def validate_production_security(self) -> None:
        """Fail fast if production environment uses insecure defaults."""
        if self.ENVIRONMENT == "production":
            insecure_key = "dev-secret-key-change-in-production-please-use-a-real-secret"
            if self.CRUCIBLE_SECRET_KEY == insecure_key or len(self.CRUCIBLE_SECRET_KEY) < 32:
                raise ValueError(
                    "CRUCIBLE_SECRET_KEY is insecure for production! "
                    "Set a random string of at least 32 characters."
                )
            if self.CRUCIBLE_API_KEY == "crucible-development-api-key":
                raise ValueError("CRUCIBLE_API_KEY must be changed from development default in production!")


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    settings = Settings()
    settings.validate_production_security()
    return settings
