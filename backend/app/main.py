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

from app.config import ensure_runtime_directories, settings
from app.database.connection import check_database_health, close_database_connections, init_db
from app.api.routes.analysis import router as analysis_router
from app.api.routes.auth import router as auth_router
from app.api.routes.speeches import router as speeches_router
from app.api.routes.users import router as users_router

LOG_LEVEL = str(settings.LOG_LEVEL or "INFO").upper()
LOG_FILE = settings.LOG_FILE or "./app.log"

log_dir = os.path.dirname(LOG_FILE)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(LOG_FILE))
except Exception:
    pass

logging.basicConfig(
    level=LOG_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=handlers,
)

logger = logging.getLogger(__name__)


def _init_confidence_calibration() -> None:
    calib_path = os.getenv("CALIBRATOR_PATH", "calibration/calibrator.json")
    try:
        if not os.path.exists(calib_path):
            logger.info("Confidence calibrator not found at: %s (skipping calibration)", calib_path)
            return

        from app.services.evaluation.calibrate import ConfidenceCalibrator, configure_calibrator

        calibrator = ConfidenceCalibrator.from_file(calib_path)
        configure_calibrator(calibrator)

        logger.info("✓ Confidence calibrator loaded successfully from: %s", calib_path)
    except ImportError as e:
        logger.warning("Calibration module not available: %s", e)
    except Exception as e:
        logger.warning("Confidence calibration init failed: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_runtime_directories()
        os.makedirs("media/uploads", exist_ok=True)

        init_db()

        health = check_database_health()
        if health.get("status") != "healthy":
            logger.error("Database health check failed: %s", health.get("error"))

        _init_confidence_calibration()

        logger.info("=" * 60)
        logger.info("PDAnalyzer API Starting")
        logger.info("=" * 60)
        logger.info("Environment: %s", settings.ENVIRONMENT)
        logger.info("API Host: %s:%s", settings.API_HOST, settings.API_PORT)
        logger.info("Database: %s", "Connected" if settings.DATABASE_URL else "Not configured")
        logger.info("Embedding Backend: %s", settings.EMBEDDING_BACKEND)
        logger.info("Default LLM: %s/%s", settings.DEFAULT_LLM_PROVIDER, settings.DEFAULT_LLM_MODEL)
        logger.info("Research Analysis: %s", "Enabled" if settings.ENABLE_RESEARCH_ANALYSIS else "Disabled")
        logger.info("=" * 60)

        yield
    except Exception as e:
        logger.error("Startup failed: %s", e, exc_info=True)
        raise
    finally:
        try:
            logger.info("Shutting down...")
            close_database_connections()
        except Exception as e:
            logger.error("Shutdown error: %s", e, exc_info=True)


app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced political speech analysis with LLM-powered discourse analysis",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

if str(settings.ENVIRONMENT or "").lower() in ("production", "prod") and ProxyHeadersMiddleware is not None:
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

os.makedirs("media", exist_ok=True)
os.makedirs("media/uploads", exist_ok=True)
app.mount("/media", StaticFiles(directory="media"), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type", "Authorization"],
    max_age=86400,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info("%s %s - Status: %s - Duration: %.3fs", request.method, request.url.path, response.status_code, duration)
    return response


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


app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(speeches_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")


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
            content={"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat() + "Z"},
        )


@app.get("/info")
async def system_info():
    try:
        from app.database.connection import get_database_stats

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
                    "research_analysis": settings.ENABLE_RESEARCH_ANALYSIS,
                    "research_llm": f"{settings.RESEARCH_LLM_PROVIDER}/{settings.RESEARCH_LLM_MODEL}",
                },
                "statistics": stats,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("Info endpoint failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat() + "Z"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=str(settings.LOG_LEVEL or "info").lower(),
        access_log=True,
    )