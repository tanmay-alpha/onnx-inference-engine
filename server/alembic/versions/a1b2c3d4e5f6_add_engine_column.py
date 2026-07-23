"""Alembic migration — add engine column to inference_logs.

Records which inference engine was used (onnxruntime, crucible, numpy, wasm).
"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = 'da7b79bcd90d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('inference_logs', sa.Column('engine', sa.String(100), nullable=False, server_default='unknown'))
    op.create_index(op.f('ix_inference_logs_engine'), 'inference_logs', ['engine'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_inference_logs_engine'), table_name='inference_logs')
    op.drop_column('inference_logs', 'engine')
