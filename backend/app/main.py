# backend/app/main.py
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# Prefer Starlette's ProxyHeadersMiddleware if available; fall back to Uvicorn; else disable
ProxyHeadersMiddleware = None
try:
    from starlette.middleware.proxy_headers import ProxyHeadersMiddleware as _StarlettePHM  # type: ignore

    ProxyHeadersMiddleware = _StarlettePHM
except Exception:
    try:
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware as _UvicornPHM  # type: ignore

        ProxyHeadersMiddleware = _UvicornPHM
    except Exception:
        ProxyHeadersMiddleware = None

# Optional: restrict Host header in production (set your domains if you enable it)
# from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import settings, ensure_runtime_directories
from app.database import check_database_health, close_database_connections, init_db

# Routers
from app.api.routes.auth import router as auth_router
from app.api.routes.users import router as users_router
from app.api.routes.speeches import router as speeches_router
from app.api.routes.analysis import router as analysis_router

# ───────────────────────────────────────────────────────────────────────────────
# Logging setup
# ───────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = settings.LOG_LEVEL.upper()
LOG_FILE = settings.LOG_FILE or "./app.log"

log_dir = os.path.dirname(LOG_FILE)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ───────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 80)
    logger.info("STARTING POLITICAL DISCOURSE ANALYSIS API")
    logger.info("=" * 80)
    try:
        # Create required runtime directories (uploads, chroma, media, etc.)
        ensure_runtime_directories()
        # Also ensure public media mount exists if you keep it exposed
        os.makedirs("media/uploads", exist_ok=True)
        logger.info("[ok] Required directories created/verified")

        # Initialize DB
        logger.info("Initializing database...")
        init_db()
        logger.info("[ok] Database initialized")

        # DB health check
        health = check_database_health()
        if health.get("status") == "healthy":
            logger.info("[ok] Database health check passed")
            logger.info("  - Database: %s", health.get("database", "Unknown"))
            logger.info("  - Version: %s", health.get("version", "Unknown"))
            logger.info("  - Response time: %s", health.get("response_time", "Unknown"))
        else:
            logger.error("Database health check failed: %s", health.get("error"))

        # Non-sensitive config echo
        logger.info("Configuration:")
        logger.info("  - Environment: %s", settings.ENVIRONMENT)
        logger.info("  - API Host: %s", settings.API_HOST)
        logger.info("  - API Port: %s", settings.API_PORT)
        logger.info("  - DATABASE_URL set: %s", bool(settings.DATABASE_URL))
        logger.info("  - Embedding Backend: %s", settings.EMBEDDING_BACKEND)
        logger.info("  - Default LLM: %s/%s", settings.DEFAULT_LLM_PROVIDER, settings.DEFAULT_LLM_MODEL)

        logger.info("=" * 80)
        logger.info("API READY - Listening for requests")
        logger.info("=" * 80)

        yield
    except Exception as e:
        logger.error("Startup failed: %s", e, exc_info=True)
        raise
    finally:
        logger.info("=" * 80)
        logger.info("SHUTTING DOWN API")
        logger.info("=" * 80)
        try:
            logger.info("Closing database connections...")
            close_database_connections()
            logger.info("[ok] Database connections closed")
        except Exception as e:
            logger.error("Shutdown error: %s", e, exc_info=True)


# ───────────────────────────────────────────────────────────────────────────────
# App
# ───────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced political speech analysis",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# If you are behind a proxy (Render/Railway), this preserves client IP info.
# Only enable in production to reduce surprises during local dev.
if settings.ENVIRONMENT.lower() in ("production", "prod") and ProxyHeadersMiddleware is not None:
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Optional: restrict Host header (set your domains if enabling it)
# if settings.ENVIRONMENT.lower() in ("production", "prod"):
#     app.add_middleware(TrustedHostMiddleware, allowed_hosts=[
#         "your-domain.com", "*.onrender.com", "*.railway.app", "*.vercel.app", "localhost"
#     ])

# Public static/media mount (remove if you don't need to expose /media)
app.mount("/media", StaticFiles(directory="media"), name="media")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,  # CSV or JSON array from env
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type", "Authorization"],
    max_age=86400,
)

# gzip
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ───────────────────────────────────────────────────────────────────────────────
# Request logging middleware
# ───────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()

    logger.info(
        "%s %s - Status: %s - Duration: %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ───────────────────────────────────────────────────────────────────────────────
# Exception handlers
# ───────────────────────────────────────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning("HTTP %s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc) if LOG_LEVEL == "DEBUG" else "An error occurred",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


# ───────────────────────────────────────────────────────────────────────────────
# Routers (mounted under /api to keep endpoints consistent)
# ───────────────────────────────────────────────────────────────────────────────
app.include_router(auth_router,     prefix="/api")
app.include_router(users_router,    prefix="/api")
app.include_router(speeches_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")

logger.info("Routes registered: /api/auth, /api/users, /api/speeches, /api/analysis")



# ───────────────────────────────────────────────────────────────────────────────
# Basic endpoints
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "auth": "/api/auth",
            "users": "/api/users",
            "speeches": "/api/speeches",
            "analysis": "/api/analysis",
            "health": "/health",
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/health")
async def health_check():
    try:
        db_health = check_database_health()
        overall_status = "healthy" if db_health.get("status") == "healthy" else "degraded"

        return {
            "status": overall_status,
            "api": {"status": "operational", "version": settings.APP_VERSION},
            "database": db_health,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )


@app.get("/info")
async def system_info():
    try:
        from app.database import get_database_stats

        stats = get_database_stats()

        return {
            "success": True,
            "data": {
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
                "database": {"database_url_set": bool(settings.DATABASE_URL)},
                "configuration": {
                    "max_speeches": settings.DEFAULT_MAX_SPEECHES,
                    "max_file_size_mb": settings.DEFAULT_MAX_FILE_SIZE / 1_000_000,
                    "embedding_backend": settings.EMBEDDING_BACKEND,
                    "default_llm": f"{settings.DEFAULT_LLM_PROVIDER}/{settings.DEFAULT_LLM_MODEL}",
                },
                "statistics": stats,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("Info endpoint failed: %s", e, exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
