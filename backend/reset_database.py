
# backend/reset_database.py
"""
DATABASE RESET UTILITY (PostgreSQL + SQLite)
===========================================

USAGE
=====

# Show database status
python reset_database.py --status

# Initialize fresh database (create tables only)
python reset_database.py --init

# Reset database (drop + recreate tables)
python reset_database.py --reset
python reset_database.py --reset --yes   # skip confirmation

# Create test data (development)
python reset_database.py --test-data
# or (alias)
python reset_database.py --seed

# Create/restore backup (PostgreSQL requires pg_dump/pg_restore in PATH)
python reset_database.py --backup
python reset_database.py --restore backups/backup_mydb_YYYYmmdd_HHMMSS.dump

NOTES
=====
- Loads .env BEFORE importing app modules so DATABASE_URL is applied.
- SQLite uses PRAGMA for schema introspection; PostgreSQL uses information_schema.
- PostgreSQL reset uses DROP SCHEMA public CASCADE + recreate schema public.
- IDEOLOGY POLICY (NEW):
  - Third family is "Centrist" (Neutral removed entirely).
  - Analyses table uses centrist_score (not neutral_score).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# IMPORTANT: load .env BEFORE importing app.database so engine uses correct DATABASE_URL
load_dotenv()

# Make sure imports work when running from backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import inspect, text

try:
    from app.database import (
        engine,
        SessionLocal,
        init_db,
        check_database_health,
        get_database_stats,
    )
except ImportError as e:
    print(f"Failed to import application modules: {e}")
    print("Make sure you're running from the backend directory")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Console helpers
# -----------------------------------------------------------------------------

class Colors:
    HEADER = "\033[95m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text_: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text_.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(text_: str) -> None:
    print(f"{Colors.OKGREEN}[OK] {text_}{Colors.ENDC}")


def print_warning(text_: str) -> None:
    print(f"{Colors.WARNING}[!] {text_}{Colors.ENDC}")


def print_error(text_: str) -> None:
    print(f"{Colors.FAIL}[X] {text_}{Colors.ENDC}")


def print_info(text_: str) -> None:
    print(f"{Colors.OKCYAN}[i] {text_}{Colors.ENDC}")


# -----------------------------------------------------------------------------
# DB type detection
# -----------------------------------------------------------------------------

def _dialect_name() -> str:
    try:
        return (engine.dialect.name or "").lower()
    except Exception:
        return ""


def _is_sqlite() -> bool:
    return _dialect_name() == "sqlite"


def _is_postgres() -> bool:
    return _dialect_name() in ("postgresql", "postgres")


def _db_label() -> str:
    if _is_postgres():
        return "PostgreSQL"
    if _is_sqlite():
        return "SQLite"
    return engine.dialect.name


# -----------------------------------------------------------------------------
# Connection info helpers
# -----------------------------------------------------------------------------

def get_database_connection_info() -> Dict[str, str]:
    """
    Best-effort connection info.
    - For SQLite, returns file path when possible.
    - For PostgreSQL, parses DATABASE_URL when available.
    """
    database_url = (os.getenv("DATABASE_URL") or "").strip()

    if _is_sqlite():
        if database_url.startswith("sqlite:///"):
            return {
                "database": database_url.replace("sqlite:///", ""),
                "host": "local",
                "port": "",
                "username": "",
                "password": "",
            }
        return {"database": database_url or "sqlite", "host": "local", "port": "", "username": "", "password": ""}

    # Postgres URL parse (best effort)
    if database_url.startswith("postgresql://"):
        # postgresql://user:pass@host:port/db
        try:
            rest = database_url.replace("postgresql://", "", 1)
            userpass, hostdb = rest.split("@", 1)
            hostport, dbname = hostdb.split("/", 1)

            if ":" in userpass:
                username, password = userpass.split(":", 1)
            else:
                username, password = userpass, ""

            if ":" in hostport:
                host, port = hostport.split(":", 1)
            else:
                host, port = hostport, "5432"

            return {
                "username": username or "postgres",
                "password": password or "",
                "host": host or "localhost",
                "port": port or "5432",
                "database": dbname or "pda",
            }
        except Exception:
            pass

    # Fallback env vars
    return {
        "username": os.getenv("DB_USERNAME", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "pda"),
    }


# -----------------------------------------------------------------------------
# Schema / row count
# -----------------------------------------------------------------------------

def _table_row_count(table: str) -> Any:
    try:
        with SessionLocal() as db:
            res = db.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
            return res.scalar()
    except Exception:
        return "?"


def _print_table_schema_sqlite(table: str) -> None:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
        print(f"\n  {table}:")
        for r in rows:
            name = r[1]
            coltype = r[2]
            notnull = "NOT NULL" if int(r[3] or 0) == 1 else "NULL"
            pk = " PK" if int(r[5] or 0) == 1 else ""
            print(f"    - {name}: {coltype} ({notnull}){pk}")
    except Exception as e:
        print_warning(f"  {table}: Error reading schema - {e}")


def _print_table_schema_postgres(table: str) -> None:
    try:
        with SessionLocal() as db:
            rows = db.execute(
                text(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :t
                    ORDER BY ordinal_position
                    """
                ),
                {"t": table},
            ).fetchall()

        print(f"\n  {table}:")
        for col_name, data_type, is_nullable in rows:
            null_info = "NULL" if is_nullable == "YES" else "NOT NULL"
            print(f"    - {col_name}: {data_type} ({null_info})")
    except Exception as e:
        print_warning(f"  {table}: Error reading schema - {e}")


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def show_database_status() -> bool:
    print_header("DATABASE STATUS")

    try:
        info = get_database_connection_info()

        print(f"{Colors.BOLD}Connection:{Colors.ENDC}")
        print(f"  Dialect: {_db_label()}")
        print(f"  DATABASE_URL set: {bool((os.getenv('DATABASE_URL') or '').strip())}")
        if _is_sqlite():
            print(f"  Database file: {info.get('database', 'sqlite')}")
        else:
            print(f"  Host: {info.get('host')}:{info.get('port')}")
            print(f"  Database: {info.get('database')}")
            print(f"  Username: {info.get('username')}")
        print()

        print(f"{Colors.BOLD}Health Check:{Colors.ENDC}")
        health = check_database_health()
        if health.get("status") == "healthy":
            print_success("Database is healthy")
            print(f"  Database: {health.get('database', 'Unknown')}")
            print(f"  Version: {health.get('version', 'Unknown')}")
            print(f"  Response time: {health.get('response_time', 'Unknown')}")
        else:
            print_error(f"Database is unhealthy: {health.get('error')}")
            return False
        print()

        print(f"{Colors.BOLD}Tables:{Colors.ENDC}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if tables:
            print_success(f"Found {len(tables)} tables:")
            for t in sorted(tables):
                print(f"  - {t}: {_table_row_count(t)} rows")
        else:
            print_warning("No tables found (database not initialized)")
        print()

        if tables:
            print(f"{Colors.BOLD}Table Details:{Colors.ENDC}")
            for t in sorted(tables):
                if _is_sqlite():
                    _print_table_schema_sqlite(t)
                else:
                    _print_table_schema_postgres(t)
            print()

        if tables:
            print(f"{Colors.BOLD}Statistics:{Colors.ENDC}")
            stats = get_database_stats()
            if isinstance(stats, dict) and "error" not in stats:
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            else:
                print_warning(f"Stats not available: {stats.get('error') if isinstance(stats, dict) else stats}")
            print()

        return True

    except Exception as e:
        print_error(f"Failed to get database status: {e}")
        return False


def initialize_database() -> bool:
    print_header("INITIALIZE DATABASE")
    try:
        print_info("Creating database tables...")
        init_db()
        print_success("Database initialized successfully")

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print_success(f"Created {len(tables)} tables:")
        for t in sorted(tables):
            print(f"  - {t}")
        return True
    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        return False


def _drop_all_tables_sqlite() -> bool:
    try:
        print_info("Dropping all SQLite tables...")
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys=OFF"))
            tables = inspect(engine).get_table_names()
            for t in tables:
                conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
                print_info(f"  Dropped table: {t}")
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()
        print_success("All SQLite tables dropped successfully")
        return True
    except Exception as e:
        print_error(f"Failed to drop SQLite tables: {e}")
        return False


def _reset_postgres_schema_public() -> bool:
    """
    Strongest reset: drop schema public cascade + recreate schema public.
    This avoids issues with enums, sequences, FK dependencies, etc.
    """
    try:
        print_info("Resetting PostgreSQL schema public (DROP SCHEMA public CASCADE)...")
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE;"))
            conn.execute(text("CREATE SCHEMA public;"))
            conn.commit()
        print_success("PostgreSQL schema public reset successfully")
        return True
    except Exception as e:
        print_error(f"Failed to reset PostgreSQL schema: {e}")
        return False


def reset_database(confirm: bool = False) -> bool:
    print_header(f"RESET DATABASE ({_db_label()})")

    # Extra safety in production
    env = (os.getenv("ENVIRONMENT") or "development").lower()
    if env in ("production", "prod"):
        print_warning("You are attempting to reset a PRODUCTION database.")
        if not confirm:
            print_error("Aborting. Use --yes to force, and be absolutely sure you are on the correct DB.")
            return False

    if not confirm:
        print_warning("WARNING: This will DELETE ALL DATA in the database!")
        print_warning("This action CANNOT be undone!")
        print()
        info = get_database_connection_info()
        if _is_sqlite():
            print_warning(f"Database (SQLite): {info.get('database')}")
        else:
            print_warning(f"Database (Postgres): {info.get('database')}")
        print()
        response = input("Type 'YES' to confirm: ").strip().upper()
        if response != "YES":
            print_info("Reset cancelled")
            return False

    try:
        if _is_postgres():
            ok = _reset_postgres_schema_public()
        else:
            ok = _drop_all_tables_sqlite()

        if not ok:
            return False

        print_info("Creating fresh tables...")
        init_db()
        print_success("Fresh database created")

        tables = inspect(engine).get_table_names()
        print_success(f"Created {len(tables)} tables:")
        for t in sorted(tables):
            print(f"  - {t}")

        return True

    except Exception as e:
        print_error(f"Failed to reset database: {e}")
        return False


def create_database_backup() -> bool:
    """
    PostgreSQL backup via pg_dump.
    SQLite backup copies db file.
    """
    print_header("CREATE DATABASE BACKUP")

    try:
        if _is_sqlite():
            info = get_database_connection_info()
            db_path = info.get("database", "")
            if not db_path:
                print_error("SQLite DATABASE_URL not set or could not determine file path.")
                return False

            src = Path(db_path)
            if not src.exists():
                print_error(f"SQLite database file not found: {src}")
                return False

            backup_dir = Path("./backups")
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = backup_dir / f"backup_sqlite_{timestamp}_{src.name}"
            dst.write_bytes(src.read_bytes())

            print_success(f"SQLite backup created: {dst}")
            return True

        # PostgreSQL
        conn_info = get_database_connection_info()
        backup_dir = Path("./backups")
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{conn_info['database']}_{timestamp}.dump"

        print_info(f"Creating PostgreSQL backup: {backup_file}")

        cmd = [
            "pg_dump",
            "-h", conn_info["host"],
            "-p", conn_info["port"],
            "-U", conn_info["username"],
            "-d", conn_info["database"],
            "-f", str(backup_file),
            "-F", "c",
            "-v",
        ]

        env = os.environ.copy()
        env["PGPASSWORD"] = conn_info.get("password", "")

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print_success(f"Backup created successfully: {backup_file}")
            return True

        print_error(f"Backup failed: {result.stderr}")
        return False

    except FileNotFoundError:
        print_error("pg_dump not found. Please install PostgreSQL client tools or add them to PATH.")
        return False
    except Exception as e:
        print_error(f"Failed to create backup: {e}")
        return False


def restore_database_backup(backup_file: str) -> bool:
    """
    PostgreSQL restore via pg_restore.
    SQLite restore copies db file over (destructive).
    """
    print_header("RESTORE DATABASE FROM BACKUP")

    p = Path(backup_file)
    if not p.exists():
        print_error(f"Backup file not found: {backup_file}")
        return False

    print_warning("WARNING: This will OVERWRITE current database!")
    print_warning("All existing data will be lost!")
    print()
    response = input("Type 'YES' to confirm: ").strip().upper()
    if response != "YES":
        print_info("Restore cancelled")
        return False

    try:
        if _is_sqlite():
            info = get_database_connection_info()
            db_path = info.get("database", "")
            if not db_path:
                print_error("SQLite DATABASE_URL not set or could not determine file path.")
                return False
            Path(db_path).write_bytes(p.read_bytes())
            print_success("SQLite database restored successfully")
            return True

        conn_info = get_database_connection_info()
        cmd = [
            "pg_restore",
            "-h", conn_info["host"],
            "-p", conn_info["port"],
            "-U", conn_info["username"],
            "-d", conn_info["database"],
            "-c",
            "-v",
            str(p),
        ]

        env = os.environ.copy()
        env["PGPASSWORD"] = conn_info.get("password", "")

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print_success("Database restored successfully")
            return True

        print_error(f"Restore failed: {result.stderr}")
        return False

    except FileNotFoundError:
        print_error("pg_restore not found. Please install PostgreSQL client tools or add them to PATH.")
        return False
    except Exception as e:
        print_error(f"Failed to restore database: {e}")
        return False


def create_test_data() -> bool:
    """
    Creates minimal test data.

    POLICY (NEW):
    - Third family is Centrist (Neutral removed).
    - Analyses use centrist_score and ideology_family="Centrist".
    """
    print_header("CREATE TEST DATA")

    try:
        # Import models
        from app.database.models import User, Speech, Analysis  # type: ignore

        # Prefer service hashing; fall back to route helper; else local context
        get_password_hash = None
        try:
            # If you have this service helper
            from app.services.auth_service import get_password_hash as _svc_hash  # type: ignore
            get_password_hash = _svc_hash
        except Exception:
            try:
                # Fallback to the route helper you already have
                from app.api.routes.auth import hash_password as _route_hash  # type: ignore
                get_password_hash = _route_hash
            except Exception:
                get_password_hash = None

        # If still None, create a local passlib context
        local_pwd_context = None
        if get_password_hash is None:
            try:
                from passlib.context import CryptContext  # type: ignore
                local_pwd_context = CryptContext(
                    schemes=["sha256_crypt"], deprecated="auto", sha256_crypt__default_rounds=535000
                )
            except Exception:
                pass

        def _hash_pwd(p: str) -> str:
            if get_password_hash is not None:
                return get_password_hash(p)
            if local_pwd_context is not None:
                return local_pwd_context.hash(p)
            raise RuntimeError("No password hashing available to create test data.")

        # Check Analysis has centrist_score
        analysis_cols = set(c.name for c in Analysis.__table__.columns)
        if "centrist_score" not in analysis_cols:
            raise RuntimeError("centrist_score column is missing on Analysis model. Check models.py/migrations.")

        # Build user kwargs depending on available columns
        user_cols = set(c.name for c in User.__table__.columns)
        user_kwargs = {
            "email": "test@example.com",
            "hashed_password": _hash_pwd("Test123"),
            "username": "testuser",
            "full_name": "Test User",
            "is_active": True,
            "is_verified": True,
            "is_admin": False,
        }
        # Optional fields
        if "role" in user_cols:
            user_kwargs["role"] = "user"
        if "subscription_tier" in user_cols:
            user_kwargs["subscription_tier"] = "free"
        if "max_speeches" in user_cols:
            user_kwargs["max_speeches"] = 50
        if "max_file_size" in user_cols:
            user_kwargs["max_file_size"] = 100_000_000

        db = SessionLocal()
        try:
            print_info("Creating test user...")
            user = db.query(User).filter(User.email == "test@example.com").first()
            if user:
                print_warning("Test user already exists, skipping...")
            else:
                user = User(**user_kwargs)
                db.add(user)
                db.flush()
                print_success(f"Created test user (ID: {user.id})")

            print_info("Creating test speech...")
            speech = Speech(
                user_id=user.id,
                title="Test Political Speech",
                speaker="Test Politician",
                text="This is a test speech for development purposes. It contains sample political content for testing.",
                date=datetime.utcnow(),
                word_count=20,
                language="en",
                source_type="test",
                status="completed",
                is_public=True,
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                use_semantic_segmentation=True,
                use_semantic_scoring=True,
            )
            db.add(speech)
            db.flush()
            print_success(f"Created test speech (ID: {speech.id})")

            print_info("Creating test analysis...")
            analysis = Analysis(
                speech_id=speech.id,
                ideology_family="Centrist",
                ideology_subtype=None,
                libertarian_score=40.0,
                authoritarian_score=35.0,
                centrist_score=25.0,
                confidence_score=0.75,
                marpor_codes=["TEST"],
                full_results={
                    "speech_level": {
                        "scores": {"Libertarian": 40.0, "Authoritarian": 35.0, "Centrist": 25.0},
                        "dominant_family": "Centrist",
                        "dominant_subtype": None,
                        "confidence_score": 0.75,
                        "marpor_codes": ["TEST"],
                    }
                },
                processing_time_seconds=1.5,
                segment_count=5,
                siu_count=2,
                key_statement_count=1,
            )
            db.add(analysis)
            db.commit()

            print_success("Test data created successfully")
            print()
            print(f"{Colors.BOLD}Test Credentials:{Colors.ENDC}")
            print("  Email: test@example.com")
            print("  Password: Test123")
            print()
            return True

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    except Exception as e:
        print_error(f"Failed to create test data: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Database management utility (SQLite + PostgreSQL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--status", action="store_true", help="Show database status")
    parser.add_argument("--init", action="store_true", help="Initialize database (create tables)")
    parser.add_argument("--reset", action="store_true", help="Reset database (drop all tables and recreate)")
    parser.add_argument("--backup", action="store_true", help="Create database backup")
    parser.add_argument("--restore", type=str, metavar="FILE", help="Restore database from backup file")
    parser.add_argument("--test-data", action="store_true", help="Create minimal test data (development)")
    parser.add_argument("--seed", action="store_true", help="Alias for --test-data (create minimal test data)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts (DANGEROUS)")

    args = parser.parse_args()

    print_header(f"DATABASE MANAGEMENT UTILITY ({_db_label()})")

    # If no explicit command: show status
    if not any([args.status, args.init, args.reset, args.backup, args.restore, args.test_data, args.seed]):
        show_database_status()
        return

    success = True

    if args.status:
        success = show_database_status() and success

    if args.backup:
        success = create_database_backup() and success

    if args.restore:
        success = restore_database_backup(args.restore) and success

    if args.reset:
        success = reset_database(confirm=args.yes) and success

    if args.init:
        success = initialize_database() and success

    if args.test_data or args.seed:
        success = create_test_data() and success

    print()
    if success:
        print_success("All operations completed successfully")
        print()
        print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("1. Restart your backend server")
        print("2. Clear browser cache (Ctrl+Shift+Delete)")
        print("3. Start adding speeches via the API/frontend")
    else:
        print_error("Some operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
