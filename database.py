# Bỏ phần Settings class đi, chỉ giữ lại:
import os
from pathlib import Path
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, declarative_base
import time
import logging

# Load .env
try:
    from dotenv import load_dotenv
    root_dir = Path(__file__).resolve().parent
    cand = [root_dir / ".env", root_dir.parent / ".env"]
    for p in cand:
        if p.exists():
            load_dotenv(p)
            break
    else:
        load_dotenv()
except Exception:
    pass

Base = declarative_base()

# Schema handling
# Prefer using PostgreSQL search_path so the app can see data in
# both the configured schema and public without hard-qualifying tables.
SCHEMA = (os.getenv("DB_SCHEMA") or "trueland").strip() or None

raw_url = (os.getenv("DATABASE_URL") or "").strip().strip('"').strip("'")
if raw_url.startswith("//"):
    raw_url = "postgresql+psycopg:" + raw_url
if not raw_url:
    raise RuntimeError("DATABASE_URL is empty or invalid")

_search_path_primary = (SCHEMA if (SCHEMA and SCHEMA.lower() != "public") else "public")
_search_path = f"{_search_path_primary},public"

engine = create_engine(
    raw_url,
    pool_size=int(os.getenv("SQLALCHEMY_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("SQLALCHEMY_MAX_OVERFLOW", "10")),
    pool_timeout=int(os.getenv("SQLALCHEMY_POOL_TIMEOUT", "10")),
    pool_recycle=int(os.getenv("SQLALCHEMY_POOL_RECYCLE", "300")),
    pool_pre_ping=True,
    connect_args={
        # Keep connection alive at TCP level (psycopg2)
        "keepalives": 1,
        "keepalives_idle": int(os.getenv("PG_KEEPALIVES_IDLE", "30")),
        "keepalives_interval": int(os.getenv("PG_KEEPALIVES_INTERVAL", "10")),
        "keepalives_count": int(os.getenv("PG_KEEPALIVES_COUNT", "5")),
        # Fail fast if network broken
        "connect_timeout": int(os.getenv("PG_CONNECT_TIMEOUT", "10")),
        # Do NOT set 'options' with search_path here: Neon pooler rejects it
    },
    echo=(os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"),
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

logger = logging.getLogger("trueland")

# Resolve effective search_path order based on where data exists
def _resolve_search_path(primary_schema: str | None) -> str:
    try:
        with engine.begin() as conn:
            # Count rows explicitly qualified to avoid current search_path
            public_cnt = 0
            other_cnt = -1
            try:
                public_cnt = conn.execute(text('SELECT count(*) FROM "public".listings')).scalar() or 0
            except Exception:
                public_cnt = 0
            if primary_schema and primary_schema.lower() != "public":
                try:
                    other_cnt = conn.execute(text(f'SELECT count(*) FROM "{primary_schema}".listings')).scalar() or 0
                except Exception:
                    other_cnt = -1
            # Choose order: if public has data and other is empty/non-existent, prefer public first
            if (primary_schema and primary_schema.lower() != "public"):
                if public_cnt > 0 and (other_cnt <= 0):
                    return f"public,{primary_schema}"
                else:
                    return f"{primary_schema},public"
            else:
                return "public"
    except Exception:
        # Fallback to configured order
        return _search_path

_resolved_search_path = _resolve_search_path(SCHEMA)
logger.info(f"Using PostgreSQL search_path: {_resolved_search_path}")

@event.listens_for(engine, "connect")
def _set_search_path(dbapi_connection, connection_record):
    try:
        cur = dbapi_connection.cursor()
        cur.execute(f"SET search_path TO {_resolved_search_path}")
        cur.close()
    except Exception:
        pass

# Ensure schema exists if not using public
try:
    if SCHEMA and SCHEMA.lower() != "public":
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
            # Also set search_path for current session (defensive, connect_args already handles it)
            conn.execute(text(f"SET search_path TO {_search_path}"))
except Exception as _e:
    logger.warning(f"Could not ensure schema '{SCHEMA}': {_e}")

# Database query logging
from sqlalchemy import event

@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())
    logger.debug(f"Query Start: {statement[:100]}...")

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if total > 1.0:
        logger.warning(f"SLOW QUERY ({total:.3f}s): {statement[:200]}")

# ❌ XÓA PHẦN Settings CLASS (không cần thiết)
