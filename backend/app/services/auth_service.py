import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import User

PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER") or None
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE") or None

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

ENVIRONMENT = (os.getenv("ENVIRONMENT") or "development").lower()
if not JWT_SECRET_KEY:
    if ENVIRONMENT in ("production", "prod"):
        raise RuntimeError("JWT secret is missing in production (set SECRET_KEY/JWT_SECRET_KEY).")
    JWT_SECRET_KEY = "dev-secret-change-this"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return PWD_CONTEXT.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return PWD_CONTEXT.hash(password)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _encode(payload: Dict[str, Any]) -> str:
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode(token: str) -> Dict[str, Any]:
    options = {"verify_aud": bool(JWT_AUDIENCE), "verify_iss": bool(JWT_ISSUER)}
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
        options=options,
    )


def _build_token_payload(
    subject: str,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = _utcnow()
    payload: Dict[str, Any] = {
        "sub": subject,
        "typ": token_type,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    if JWT_ISSUER:
        payload["iss"] = JWT_ISSUER
    if JWT_AUDIENCE:
        payload["aud"] = JWT_AUDIENCE
    if extra_claims:
        for k, v in extra_claims.items():
            if k not in payload:
                payload[k] = v
    return payload


def issue_token_pair(subject: str, extra_claims: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    access_payload = _build_token_payload(
        subject=subject,
        token_type="access",
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        extra_claims=extra_claims,
    )
    refresh_payload = _build_token_payload(
        subject=subject,
        token_type="refresh",
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        extra_claims=extra_claims,
    )
    return {
        "access_token": _encode(access_payload),
        "refresh_token": _encode(refresh_payload),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


def refresh_tokens(refresh_token: str) -> Dict[str, Any]:
    try:
        payload = _decode(refresh_token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    if payload.get("typ") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    reserved = {"sub", "typ", "jti", "iat", "nbf", "exp", "iss", "aud"}
    carried_claims = {k: v for k, v in payload.items() if k not in reserved}
    return issue_token_pair(subject=str(subject), extra_claims=carried_claims)


def authenticate_user(db: Session, identifier: str, password: str) -> Optional[User]:
    ident = (identifier or "").strip()
    if not ident:
        return None

    q = db.query(User).filter(or_(User.email == ident, User.username == ident))
    user = q.first()
    if not user:
        return None

    hashed = getattr(user, "hashed_password", None)
    if not hashed:
        return None

    if not verify_password(password, hashed):
        return None

    if hasattr(user, "is_active") and not bool(user.is_active):
        return None

    return user


def _resolve_subject_to_user(db: Session, sub: Union[str, int]) -> Optional[User]:
    s = str(sub).strip()
    if not s:
        return None

    if s.isdigit():
        return db.query(User).filter(User.id == int(s)).first()

    return db.query(User).filter(or_(User.email == s, User.username == s)).first()


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = _decode(token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

    token_type = payload.get("type") or payload.get("typ")
    if token_type != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

    user = _resolve_subject_to_user(db, sub)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if hasattr(user, "is_active") and not bool(user.is_active):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return user