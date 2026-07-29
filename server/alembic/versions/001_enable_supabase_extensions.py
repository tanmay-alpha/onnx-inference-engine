"""Enable Supabase extensions.

Supabase ships with several PostgreSQL extensions pre-installed but they must
be explicitly enabled per-database.  This migration enables the three most
commonly used ones:

  pgcrypto           – UUID generation, password hashing
  pg_stat_statements – Query performance tracking
  vector             – pgvector (for future semantic / embedding search)

Run with:
    cd server && alembic upgrade head
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "001_enable_supabase_extensions"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Enable required Supabase extensions."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
    # Uncomment when ready for vector / embedding search:
    # op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    """Disable extensions (kept as no-op for safety)."""
    # op.execute("DROP EXTENSION IF EXISTS pg_stat_statements;")
    # op.execute("DROP EXTENSION IF EXISTS pgcrypto;")
    # op.execute("DROP EXTENSION IF EXISTS vector;")
    pass
