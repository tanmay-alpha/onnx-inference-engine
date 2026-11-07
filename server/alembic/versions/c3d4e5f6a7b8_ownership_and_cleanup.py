"""Ownership columns; drop the benchmarks table.

* fraud_cases.user_id — fraud checks are scoped to the submitting user.
* api_keys.key_prefix — real key prefix for display (the hash was shown before).
* benchmarks — removed: benchmark numbers are published as a static,
  reproducible results.json, not stored from unauthenticated POSTs.
"""
from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6a7b8"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("fraud_cases") as batch:
        batch.add_column(sa.Column("user_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key("fk_fraud_cases_user_id_users", "users", ["user_id"], ["id"],
                                 ondelete="SET NULL")
        batch.create_index("ix_fraud_cases_user_id", ["user_id"], unique=False)
    with op.batch_alter_table("api_keys") as batch:
        batch.add_column(sa.Column("key_prefix", sa.String(length=16), nullable=True))
    op.drop_index(op.f("ix_benchmarks_engine"), table_name="benchmarks")
    op.drop_index(op.f("ix_benchmarks_created_at"), table_name="benchmarks")
    op.drop_table("benchmarks")


def downgrade() -> None:
    op.create_table(
        "benchmarks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("engine", sa.String(length=100), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("memory_mb", sa.Float(), nullable=True),
        sa.Column("device", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_benchmarks_created_at"), "benchmarks", ["created_at"], unique=False)
    op.create_index(op.f("ix_benchmarks_engine"), "benchmarks", ["engine"], unique=False)
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("key_prefix")
    with op.batch_alter_table("fraud_cases") as batch:
        batch.drop_index("ix_fraud_cases_user_id")
        batch.drop_constraint("fk_fraud_cases_user_id_users", type_="foreignkey")
        batch.drop_column("user_id")
