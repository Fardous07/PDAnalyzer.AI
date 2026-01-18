# backend/app/database/__init__.py

"""Database package exports."""

from app.database.connection import (
    Base,
    engine,
    SessionLocal,
    get_db,
    get_db_context,
    init_db,
    drop_db,
    reset_db,
    check_database_health,
    get_database_stats,
    connect_with_retry,
    close_database_connections,
    get_table_names,
    vacuum_database,
    backup_sqlite_database,
)

from app.database.models import (
    User,
    Speech,
    Analysis,
    Question,
    Project,
    AnalysisHistory,
    speech_project_association,
)

__all__ = [
    # Connection
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

    # Models
    "User",
    "Speech",
    "Analysis",
    "Question",
    "Project",
    "AnalysisHistory",
    "speech_project_association",
]
