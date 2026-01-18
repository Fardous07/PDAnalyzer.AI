"""
backend/app/api/utils/responses.py

Standard API response helpers for DiscourseAI Backend.
Keeps a uniform envelope across endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def _now_z() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_response(
    *,
    success: bool,
    data: Any = None,
    message: str = "",
    error: Optional[Any] = None,
    processing_time: Optional[float] = None,
    timestamp: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standard response envelope.

    Shape:
    {
      "success": bool,
      "data": any,
      "message": str,
      "error": any | null,
      "processing_time": float | null,
      "timestamp": str,
      "meta": { ... } | null
    }
    """
    return {
        "success": bool(success),
        "data": data,
        "message": message or "",
        "error": error,
        "processing_time": processing_time,
        "timestamp": timestamp or _now_z(),
        "meta": meta,
    }


__all__ = ["create_response"]
