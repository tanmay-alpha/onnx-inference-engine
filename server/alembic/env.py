"""Alembic environment configuration for Crucible."""
from logging.config import fileConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from alembic import context

import os
import sys
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Add project root directory to path so 'from server.xxx' imports work
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from server.database import Base
from server.models import User, ApiKey, ModelRecord, InferenceLog, FraudCase, Benchmark  # noqa: F401
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_db_url() -> str:
    """Get the async database URL, preferring the env var from the project root."""
    # Check DATABASE_URL from environment (project root .env sets this)
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # Fallback to alembic.ini value
    return config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection | AsyncConnection) -> None:
    """Run migrations on a live connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def _prep_name_func():
    import uuid
    return f"__asyncpg_stmt_{uuid.uuid4().hex}__"


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = get_db_url()
    connect_args = {}
    pool_kwargs = {}
    if "postgresql" in url:
        from sqlalchemy.pool import NullPool
        pool_kwargs["poolclass"] = NullPool
        connect_args = {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "prepared_statement_name_func": _prep_name_func,
        }
    connectable = create_async_engine(url, connect_args=connect_args, **pool_kwargs)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())
