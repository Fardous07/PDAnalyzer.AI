# app/services/auth_service.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import hashlib
import logging

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database.models import User
from app.database.connection import get_db

logger = logging.getLogger(__name__)

# Try to import settings, provide defaults if it fails
try:
    from app.config import settings
except ImportError:
    logger.warning("Could not import settings from app.config. Using defaults.")
    # Create a simple settings object with defaults
    class SimpleSettings:
        SECRET_KEY = "SECRET_KEY"
        ALGORITHM = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    settings = SimpleSettings()

# Use SHA256-Crypt for password hashing (no 72-byte bcrypt truncation issue)
pwd_context = CryptContext(
    schemes=["sha256_crypt"],
    deprecated="auto",
    sha256_crypt__default_rounds=535000,
)

# IMPORTANT: leading slash improves OpenAPI/Swagger + client compatibility
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Custom fallback prefix used in get_password_hash() if passlib fails
_FALLBACK_PREFIX = "sha256$"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against:
    1) passlib sha256_crypt hash (preferred), OR
    2) fallback "sha256$<hex>" hash (only if passlib hashing failed earlier)
    """
    try:
        if not plain_password or not hashed_password:
            return False

        # Handle fallback hashes created by get_password_hash() on rare failures
        if hashed_password.startswith(_FALLBACK_PREFIX):
            expected = hashed_password[len(_FALLBACK_PREFIX) :]
            actual = _sha256_hex(plain_password)
            # constant-time compare
            return hashlib.compare_digest(expected, actual)

        # Normal path: passlib verification
        return pwd_context.verify(plain_password, hashed_password)

    except Exception as e:
        logger.warning(f"Password verification failed: {str(e)}")
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password with sha256_crypt.
    If passlib fails (misconfig / environment), fall back to "sha256$<hex>".
    """
    try:
        if not password:
            raise ValueError("Empty password not allowed")
        return pwd_context.hash(password)

    except Exception as e:
        logger.error(f"Password hashing failed (passlib). Using fallback sha256: {str(e)}")
        return f"{_FALLBACK_PREFIX}{_sha256_hex(password)}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    EXPECTATION:
    - Caller includes "sub" (subject) in data, typically str(user.id).
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=getattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 30))

    to_encode.update({"exp": expire})
    
    # CRITICAL FIX: Ensure 'sub' is always a string (JWT requirement)
    if "sub" in to_encode and not isinstance(to_encode["sub"], str):
        to_encode["sub"] = str(to_encode["sub"])
        logger.debug(f"Converted 'sub' to string: {to_encode['sub']}")

    encoded_jwt = jwt.encode(to_encode, getattr(settings, 'SECRET_KEY', 'SECRET_KEY'), 
                             algorithm=getattr(settings, 'ALGORITHM', 'HS256'))
    
    logger.debug(f"Created JWT token for sub: {to_encode.get('sub')}")
    return encoded_jwt


def authenticate_user(db: Session, email_or_username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by email OR username, plus password.
    Returns the User if valid, else None.
    """
    if not email_or_username or not password:
        return None

    user = db.query(User).filter(User.email == email_or_username).first()
    if not user:
        user = db.query(User).filter(User.username == email_or_username).first()

    if not user:
        logger.warning(f"User not found: {email_or_username}")
        return None

    if not verify_password(password, user.hashed_password):
        logger.warning(f"Password verification failed for user: {email_or_username}")
        return None

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    logger.info(f"User authenticated: {email_or_username} (id: {user.id})")
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency: get current user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        secret_key = getattr(settings, 'SECRET_KEY', 'SECRET_KEY')
        logger.debug(f"Decoding JWT with secret key: {secret_key[:10]}...")
        
        payload = jwt.decode(token, secret_key, 
                            algorithms=[getattr(settings, 'ALGORITHM', 'HS256')])
        
        user_id = payload.get("sub")
        logger.debug(f"JWT payload decoded - sub: {user_id}, type: {type(user_id)}")
        
        if user_id is None:
            logger.warning("No 'sub' claim in JWT token")
            raise credentials_exception

        # CRITICAL FIX: Convert user_id to int whether it's string or number
        try:
            user_id_int = int(user_id)  # This works for both "123" and 123
            logger.debug(f"Converted user_id to int: {user_id_int}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid user_id format in JWT: {user_id} (error: {e})")
            raise credentials_exception

    except JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id_int).first()
    if user is None:
        logger.error(f"User not found with id: {user_id_int}")
        raise credentials_exception

    if not user.is_active:
        logger.warning(f"User {user_id_int} is not active")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    logger.info(f"Current user retrieved: {user.email} (id: {user.id})")
    return user


def create_user(db: Session, user_data: dict) -> User:
    """
    Create a new user with hashed password.
    Expects user_data keys: email, username, password, optional full_name, organization.
    """
    # Check if email exists
    existing_email = db.query(User).filter(User.email == user_data["email"]).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Check if username exists
    existing_username = db.query(User).filter(User.username == user_data["username"]).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    hashed_password = get_password_hash(user_data["password"])
    logger.info(f"Password hashed successfully for user: {user_data['email']}")

    db_user = User(
        email=user_data["email"],
        username=user_data["username"],
        full_name=user_data.get("full_name"),
        organization=user_data.get("organization"),
        hashed_password=hashed_password,
        is_active=True,
        is_verified=False,
        role="user",
        subscription_tier="free",
        max_speeches=50,
        max_file_size=100_000_000,  # 100MB
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    logger.info(f"User created successfully: {db_user.email} (id: {db_user.id})")
    return db_user


def update_user_last_login(db: Session, user_id: int) -> None:
    """
    Update user's last_login timestamp.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = datetime.utcnow()
        db.commit()
        logger.debug(f"Updated last login for user_id: {user_id}")


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """
    Get user by ID.
    """
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get user by email.
    """
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """
    Get user by username.
    """
    return db.query(User).filter(User.username == username).first()


__all__ = [
    "get_password_hash",
    "verify_password",
    "create_access_token",
    "authenticate_user",
    "get_current_user",
    "create_user",
    "update_user_last_login",
    "get_user_by_id",
    "get_user_by_email",
    "get_user_by_username",
]