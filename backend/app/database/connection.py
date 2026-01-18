# backend/app/database/connection.py
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Set

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# ✅ DO NOT TOUCH .env FILE — but we load it here so any script importing this
# module gets DATABASE_URL correctly (reset_database.py included).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass

from app.database.models import Base  # single source of truth

logger = logging.getLogger(__name__)


def _mask_db_url(url: str) -> str:
    if not url:
        return ""
    if "://" in url and "@" in url:
        left, right = url.split("@", 1)
        if ":" in left:
            scheme_and_user, _ = left.rsplit(":", 1)
            return f"{scheme_and_user}:***@{right}"
    return url


# -----------------------------------------------------------------------------
# PostgreSQL (primary)
# -----------------------------------------------------------------------------
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. PostgreSQL requires DATABASE_URL.")

IS_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")
if not IS_POSTGRES:
    raise RuntimeError(f"DATABASE_URL must be PostgreSQL (postgresql://...). Got: {_mask_db_url(DATABASE_URL)}")

DB_ECHO = (os.getenv("DB_ECHO") or "false").lower() == "true"
DB_POOL_PRE_PING = (os.getenv("DB_POOL_PRE_PING") or "true").lower() == "true"

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Force UTC server-side if possible
CONNECT_ARGS: Dict[str, Any] = {"connect_timeout": 10, "options": "-c timezone=utc"}


def create_database_engine() -> Engine:
    logger.info("Creating PostgreSQL engine: %s", _mask_db_url(DATABASE_URL))
    try:
        eng = create_engine(
            DATABASE_URL,
            echo=DB_ECHO,
            pool_pre_ping=DB_POOL_PRE_PING,
            connect_args=CONNECT_ARGS,
            poolclass=QueuePool,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_recycle=DB_POOL_RECYCLE,
        )

        # Verify connectivity
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("PostgreSQL engine created and verified successfully")
        return eng

    except Exception as e:
        logger.error("Failed to create PostgreSQL engine: %s", e, exc_info=True)
        raise


engine = create_database_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Database error: %s", e, exc_info=True)
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Database error: %s", e, exc_info=True)
        raise
    finally:
        db.close()


def init_db() -> None:
    logger.info("Initializing database (creating tables)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db() -> None:
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")


def reset_db() -> None:
    logger.warning("Resetting database...")
    drop_db()
    init_db()
    logger.info("Database reset complete")


def check_database_health() -> Dict[str, Any]:
    try:
        start = time.time()
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar()
        return {
            "status": "healthy",
            "database": "PostgreSQL",
            "version": version,
            "response_time": f"{time.time() - start:.3f}s",
            "url": _mask_db_url(DATABASE_URL),
        }
    except Exception as e:
        logger.error("Database health check failed: %s", e, exc_info=True)
        return {"status": "unhealthy", "error": str(e), "url": _mask_db_url(DATABASE_URL)}


# -----------------------------------------------------------------------------
# Schema helpers (dialect-safe; avoids ORM mismatches)
# -----------------------------------------------------------------------------

def _table_columns(table: str) -> Set[str]:
    """
    Return the actual column names present in the database for a table.
    PostgreSQL via SQLAlchemy inspector.
    """
    try:
        insp = inspect(engine)
        cols = insp.get_columns(table)
        return {str(c.get("name")) for c in (cols or []) if c.get("name")}
    except Exception:
        return set()


def _safe_scalar(db: Session, sql: str, params: Optional[Dict[str, Any]] = None, default: Any = 0) -> Any:
    try:
        return db.execute(text(sql), params or {}).scalar()
    except Exception:
        return default


def get_database_stats() -> Dict[str, Any]:
    """
    Robust stats that do NOT depend on Python ORM attributes.
    - Works during schema migration (legacy neutral_score or centrist_score).
    - Uses real DB schema to decide which columns exist.
    - API-facing output is Centrist-only (no 'neutral' key).
    """
    try:
        with SessionLocal() as db:
            stats: Dict[str, Any] = {}

            stats["total_speeches"] = int(_safe_scalar(db, 'SELECT COUNT(*) FROM "speeches"', default=0) or 0)
            stats["analyzed_speeches"] = int(
                _safe_scalar(db, 'SELECT COUNT(*) FROM "speeches" s JOIN "analyses" a ON a.speech_id = s.id', default=0) or 0
            )
            stats["pending_speeches"] = int(
                _safe_scalar(db, 'SELECT COUNT(*) FROM "speeches" WHERE status = :st', {"st": "pending"}, default=0) or 0
            )
            stats["total_users"] = int(_safe_scalar(db, 'SELECT COUNT(*) FROM "users"', default=0) or 0)
            stats["active_users"] = int(_safe_scalar(db, 'SELECT COUNT(*) FROM "users" WHERE is_active = true', default=0) or 0)
            stats["total_questions"] = int(_safe_scalar(db, 'SELECT COUNT(*) FROM "questions"', default=0) or 0)
            stats["total_projects"] = int(_safe_scalar(db, 'SELECT COUNT(*) FROM "projects"', default=0) or 0)

            # Ideology distribution
            try:
                rows = db.execute(text('SELECT ideology_family, COUNT(*) FROM "analyses" GROUP BY ideology_family')).fetchall()
                stats["ideology_distribution"] = {str(r[0]): int(r[1]) for r in rows if r and r[0]}
            except Exception:
                stats["ideology_distribution"] = {}

            # Average scores: determine third column by actual DB schema
            analysis_cols = _table_columns("analyses")
            third_col = None
            if "centrist_score" in analysis_cols:
                third_col = "centrist_score"
            elif "neutral_score" in analysis_cols:
                third_col = "neutral_score"

            avg = {"libertarian": 0.0, "authoritarian": 0.0, "centrist": 0.0, "confidence": 0.0}

            if third_col:
                row = db.execute(
                    text(
                        f'''
                        SELECT
                          AVG(libertarian_score) AS avg_lib,
                          AVG(authoritarian_score) AS avg_auth,
                          AVG({third_col}) AS avg_third,
                          AVG(confidence_score) AS avg_conf
                        FROM "analyses"
                        '''
                    )
                ).fetchone()

                if row:
                    avg["libertarian"] = float(row[0] or 0.0)
                    avg["authoritarian"] = float(row[1] or 0.0)
                    avg["centrist"] = float(row[2] or 0.0)  # API label is centrist regardless of DB col name
                    avg["confidence"] = float(row[3] or 0.0)

            stats["average_scores"] = {
                "libertarian": round(avg["libertarian"], 2),
                "authoritarian": round(avg["authoritarian"], 2),
                "centrist": round(avg["centrist"], 2),
                "confidence": round(avg["confidence"], 2),
            }

            project_cols = _table_columns("projects")
            stats["schema_flags"] = {
                "dialect": engine.dialect.name,
                "database_url_effective": _mask_db_url(DATABASE_URL),
                "analyses_has_centrist_score": "centrist_score" in analysis_cols,
                "analyses_has_neutral_score": "neutral_score" in analysis_cols,
                "projects_columns": sorted(list(project_cols)),
            }

            return stats

    except Exception as e:
        logger.error("Failed to get database stats: %s", e, exc_info=True)
        return {"error": str(e)}


def connect_with_retry(max_retries: int = 5, delay_seconds: float = 2.0) -> Engine:
    last_err: Optional[Exception] = None
    delay = float(delay_seconds)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("DB connection attempt %s/%s", attempt, max_retries)
            eng = create_database_engine()
            logger.info("DB connection established")
            return eng
        except OperationalError as e:
            last_err = e
            if attempt == max_retries:
                break
            logger.warning("DB not ready, retrying in %.1fs: %s", delay, e)
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Failed to connect to database after {max_retries} attempts: {last_err}")


def close_database_connections() -> None:
    engine.dispose()
    logger.info("Database connections closed")


def get_table_names() -> list:
    try:
        return [table.name for table in Base.metadata.sorted_tables]
    except Exception as e:
        logger.error("Failed to get table names: %s", e)
        return []


def vacuum_database() -> None:
    # PostgreSQL vacuum is an admin operation; do not run automatically.
    logger.warning("VACUUM is not run from the app for PostgreSQL.")


def backup_sqlite_database(_backup_path: str) -> None:
    logger.warning("SQLite backup helper is not applicable for PostgreSQL.")


__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_context",
    "init_db",
    "drop_db",
    "reset_db",
    "check_database_health",
    "get_database_stats",
    "connect_with_retry",
    "close_database_connections",
    "get_table_names",
    "vacuum_database",
    "backup_sqlite_database",
]
