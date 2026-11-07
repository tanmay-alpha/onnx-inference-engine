"""Production settings validation."""
from __future__ import annotations

import pytest

from server.config import DEV_SECRET_KEY, Settings

STRONG = "x" * 40


def _settings(**env) -> Settings:
    return Settings(_env_file=None, **env)


def test_production_refuses_default_jwt_secret() -> None:
    s = _settings(ENVIRONMENT="production", CRUCIBLE_SECRET_KEY=DEV_SECRET_KEY,
                  DATABASE_URL="postgresql://u:p@h/db")
    with pytest.raises(ValueError, match="CRUCIBLE_SECRET_KEY"):
        s.validate_production_security()


def test_production_requires_database_url(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = _settings(ENVIRONMENT="production", CRUCIBLE_SECRET_KEY=STRONG)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        s.validate_production_security()


def test_production_needs_nothing_else() -> None:
    _settings(ENVIRONMENT="production", CRUCIBLE_SECRET_KEY=STRONG,
              DATABASE_URL="postgresql://u:p@h/db").validate_production_security()


def test_development_defaults_are_accepted() -> None:
    _settings().validate_production_security()
