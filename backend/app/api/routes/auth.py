# backend/app/api/routes/auth.py

"""
AUTHENTICATION ROUTES - JWT + User Management
==============================================
(Pydantic v2)
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator
from sqlalchemy.orm import Session

from app.config import settings

# JWT and password hashing
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext

    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False

# Database
from app.database import get_db, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


# =============================================================================
# CONFIGURATION
# =============================================================================

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS

pwd_context = (
    CryptContext(
        schemes=["sha256_crypt"],
        deprecated="auto",
        sha256_crypt__default_rounds=535000,
    )
    if _AUTH_AVAILABLE
    else None
)

security = HTTPBearer()


# =============================================================================
# REQUEST/RESPONSE MODELS (Pydantic v2)
# =============================================================================

class UserRegister(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123",
                "username": "johndoe",
                "full_name": "John Doe",
                "organization": "Acme Corp",
            }
        }
    )

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=100, description="Password (min 8 characters)")
    username: Optional[str] = Field(None, min_length=3, max_length=100, description="Username (optional)")
    full_name: Optional[str] = Field(None, max_length=200)
    organization: Optional[str] = Field(None, max_length=200)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: Optional[str]
    full_name: Optional[str]
    organization: Optional[str]
    bio: Optional[str]
    is_active: bool
    is_verified: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime]
    preferences: Optional[Dict[str, Any]]


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=200)
    organization: Optional[str] = Field(None, max_length=200)
    bio: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class PasswordReset(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class PreferencesUpdate(BaseModel):
    preferences: Dict[str, Any] = Field(..., description="User preferences (JSON)")


# =============================================================================
# PASSWORD UTILITIES
# =============================================================================

def hash_password(password: str) -> str:
    if not _AUTH_AVAILABLE or not pwd_context:
        raise RuntimeError("Password hashing not available")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not _AUTH_AVAILABLE or not pwd_context:
        raise RuntimeError("Password verification not available")
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# JWT TOKEN UTILITIES
# =============================================================================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    if not _AUTH_AVAILABLE:
        raise RuntimeError("JWT not available")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not set")

    to_encode = data.copy()

    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})

    # JWT best practice: sub as string
    if "sub" in to_encode and to_encode["sub"] is not None:
        to_encode["sub"] = str(to_encode["sub"])

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: Dict[str, Any]) -> str:
    if not _AUTH_AVAILABLE:
        raise RuntimeError("JWT not available")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not set")

    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})

    if "sub" in to_encode and to_encode["sub"] is not None:
        to_encode["sub"] = str(to_encode["sub"])

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    if not _AUTH_AVAILABLE:
        raise RuntimeError("JWT not available")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not set")

    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _coerce_sub_to_int(payload: Dict[str, Any]) -> int:
    sub = payload.get("sub")
    try:
        return int(sub)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# AUTHENTICATION DEPENDENCIES
# =============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
) -> User:
    if not _AUTH_AVAILABLE:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication not available")

    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user_id = _coerce_sub_to_int(payload)
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")

    return user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough privileges")
    return current_user


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@router.post("/register", response_model=Dict[str, Any])
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        if not _AUTH_AVAILABLE:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication not available")

        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        if user_data.username:
            existing_username = db.query(User).filter(User.username == user_data.username).first()
            if existing_username:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")

        hashed_password = hash_password(user_data.password)

        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            username=user_data.username,
            full_name=user_data.full_name,
            organization=user_data.organization,
            is_active=True,
            is_verified=False,
            is_admin=False,
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info("New user registered: %s", user.email)

        access_token = create_access_token({"sub": str(user.id), "email": user.email})
        refresh_token = create_refresh_token({"sub": str(user.id)})

        return {
            "success": True,
            "message": "User registered successfully",
            "user": UserResponse.model_validate(user).model_dump(),
            "tokens": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/login", response_model=Dict[str, Any])
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    try:
        if not _AUTH_AVAILABLE:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication not available")

        user = db.query(User).filter(User.email == credentials.email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

        if not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

        user.last_login = datetime.utcnow()
        db.commit()

        logger.info("User logged in: %s", user.email)

        access_token = create_access_token({"sub": str(user.id), "email": user.email})
        refresh_token = create_refresh_token({"sub": str(user.id)})

        return {
            "success": True,
            "message": "Login successful",
            "user": UserResponse.model_validate(user).model_dump(),
            "tokens": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh, db: Session = Depends(get_db)):
    try:
        if not _AUTH_AVAILABLE:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication not available")

        payload = decode_token(token_data.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

        user_id = _coerce_sub_to_int(payload)
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        new_access_token = create_access_token({"sub": str(user.id), "email": user.email})
        new_refresh_token = create_refresh_token({"sub": str(user.id)})

        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/logout", response_model=Dict[str, Any])
async def logout(current_user: User = Depends(get_current_user)):
    logger.info("User logged out: %s", current_user.email)
    return {"success": True, "message": "Logout successful (delete token on client)"}


# =============================================================================
# USER MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        update_data = user_update.model_dump(exclude_unset=True)

        if "username" in update_data and update_data["username"] != current_user.username:
            existing = db.query(User).filter(User.username == update_data["username"]).first()
            if existing:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")

        for field, value in update_data.items():
            setattr(current_user, field, value)

        current_user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(current_user)

        logger.info("User profile updated: %s", current_user.email)
        return UserResponse.model_validate(current_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/me/password", response_model=Dict[str, Any])
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        if not _AUTH_AVAILABLE:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication not available")

        if not verify_password(password_change.current_password, current_user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")

        new_hashed = hash_password(password_change.new_password)

        current_user.hashed_password = new_hashed
        current_user.updated_at = datetime.utcnow()
        db.commit()

        logger.info("Password changed for user: %s", current_user.email)
        return {"success": True, "message": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/me/preferences", response_model=UserResponse)
async def update_preferences(
    preferences_update: PreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        current_user.preferences = preferences_update.preferences
        current_user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(current_user)

        logger.info("Preferences updated for user: %s", current_user.email)
        return UserResponse.model_validate(current_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Preferences update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/me", response_model=Dict[str, Any])
async def delete_current_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        email = current_user.email
        db.delete(current_user)
        db.commit()

        logger.info("User account deleted: %s", email)
        return {"success": True, "message": "Account deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Account deletion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@router.get("/users", response_model=Dict[str, Any])
async def list_users(
    page: int = 1,
    page_size: int = 20,
    admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    try:
        query = db.query(User)
        total = query.count()

        offset = (page - 1) * page_size
        users = query.offset(offset).limit(page_size).all()

        return {
            "success": True,
            "data": {
                "users": [UserResponse.model_validate(u).model_dump() for u in users],
                "total": total,
                "page": page,
                "page_size": page_size,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("List users failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        return UserResponse.model_validate(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get user failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


__all__ = [
    "router",
    "get_current_user",
    "get_current_admin_user",
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
]