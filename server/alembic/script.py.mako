"""Alembic migration script template.

Each migration file is generated from this template.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import table, column

revision = '${up_revision}'
down_revision = '${down_revision}'
branch_labels = ${branch_labels}
depends_on = ${depends_on}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
