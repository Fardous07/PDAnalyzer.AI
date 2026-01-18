# backend/app/api/routes/users.py

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_, desc

from app.database import get_db, User, Speech, Analysis

# Auth dependency
try:
    from app.api.routes.auth import get_current_user, get_current_admin_user
    _AUTH_AVAILABLE = True
except Exception:
    _AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/users", tags=["users"])


# =============================================================================
# RESPONSE HELPER
# =============================================================================

def create_response(
    success: bool,
    data: Optional[Any] = None,
    error: Optional[str] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "success": success,
        "data": data,
        "error": error,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# =============================================================================
# AUTH HELPERS
# =============================================================================

async def require_current_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not _AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    return current_user


async def require_admin_user(
    current_user: User = Depends(get_current_admin_user),
) -> User:
    if not _AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    return current_user


# =============================================================================
# Pydantic Models
# =============================================================================

class PublicUserProfile(BaseModel):
    id: int
    username: Optional[str] = None
    full_name: Optional[str] = None
    organization: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime
    speech_count: int = 0
    analysis_count: int = 0

    class Config:
        from_attributes = True


class UserStats(BaseModel):
    total_speeches: int
    analyzed_speeches: int
    pending_speeches: int
    total_word_count: int
    avg_confidence: float
    ideology_distribution: Dict[str, int]
    recent_speeches_30d: int
    last_speech_date: Optional[datetime]


class UserUpdateAdmin(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=200)
    organization: Optional[str] = Field(None, max_length=200)
    bio: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_admin: Optional[bool] = None


class BulkUpdateRequest(BaseModel):
    user_ids: List[int] = Field(..., min_items=1, max_items=100)
    updates: Dict[str, Any]


class BulkDeleteRequest(BaseModel):
    user_ids: List[int] = Field(..., min_items=1, max_items=100)
    confirm: bool = Field(..., description="Must be true to confirm deletion")


# =============================================================================
# INTERNAL: Statistics computation
# =============================================================================

def _get_user_statistics(db: Session, user_id: int) -> UserStats:
    speeches_q = db.query(Speech).filter(Speech.user_id == user_id)
    total_speeches = speeches_q.count()

    analyzed_speeches = db.query(func.count(Analysis.id)).join(Speech).filter(Speech.user_id == user_id).scalar() or 0
    pending_speeches = max(0, int(total_speeches) - int(analyzed_speeches))

    total_words = db.query(func.sum(Speech.word_count)).filter(Speech.user_id == user_id).scalar() or 0
    avg_conf = db.query(func.avg(Analysis.confidence_score)).join(Speech).filter(Speech.user_id == user_id).scalar() or 0.0

    ideology_dist = (
        db.query(Analysis.ideology_family, func.count(Analysis.id))
        .join(Speech)
        .filter(Speech.user_id == user_id)
        .group_by(Analysis.ideology_family)
        .all()
    )
    ideology_dict = {str(fam): int(cnt) for fam, cnt in ideology_dist}

    cutoff = datetime.utcnow() - timedelta(days=30)
    recent_30d = speeches_q.filter(Speech.created_at >= cutoff).count()

    last_speech = speeches_q.order_by(desc(Speech.date)).first()
    last_speech_date = last_speech.date if last_speech else None

    return UserStats(
        total_speeches=int(total_speeches),
        analyzed_speeches=int(analyzed_speeches),
        pending_speeches=int(pending_speeches),
        total_word_count=int(total_words),
        avg_confidence=float(avg_conf),
        ideology_distribution=ideology_dict,
        recent_speeches_30d=int(recent_30d),
        last_speech_date=last_speech_date,
    )


# =============================================================================
# PUBLIC ROUTES
# =============================================================================

@router.get("/search", response_model=Dict[str, Any])
async def search_users(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    search_term = f"%{q.strip()}%"

    users = (
        db.query(User)
        .filter(
            and_(
                User.is_active == True,
                or_(
                    User.username.ilike(search_term),
                    User.full_name.ilike(search_term),
                    User.organization.ilike(search_term),
                ),
            )
        )
        .limit(limit)
        .all()
    )

    results: List[Dict[str, Any]] = []
    for u in users:
        public_speech_count = db.query(func.count(Speech.id)).filter(Speech.user_id == u.id, Speech.is_public == True).scalar() or 0
        results.append(
            {
                "id": u.id,
                "username": u.username,
                "full_name": u.full_name,
                "organization": u.organization,
                "speech_count": int(public_speech_count),
            }
        )

    return create_response(True, data={"users": results, "count": len(results), "query": q})


@router.get("/speakers", response_model=Dict[str, Any])
async def get_speakers(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    try:
        if _AUTH_AVAILABLE:
            from app.api.routes.auth import get_current_user
            from fastapi import Request
            # Try to get current user but don't require it
            current_user = None
        else:
            current_user = None
    except:
        current_user = None

    is_admin = bool(getattr(current_user, "is_admin", False)) if current_user is not None else False

    q = db.query(Speech.speaker, func.count(Speech.id).label("speech_count"))
    if not is_admin:
        q = q.filter(Speech.is_public == True)

    speakers = (
        q.group_by(Speech.speaker)
        .order_by(desc("speech_count"))
        .limit(limit)
        .all()
    )

    out = [{"speaker": s, "speech_count": int(c)} for s, c in speakers if s]
    return create_response(True, data={"speakers": out, "count": len(out)})


@router.get("/stats/overview", response_model=Dict[str, Any])
async def system_user_stats_overview(
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    total_users = db.query(func.count(User.id)).scalar() or 0
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar() or 0
    verified_users = db.query(func.count(User.id)).filter(User.is_verified == True).scalar() or 0
    admin_users = db.query(func.count(User.id)).filter(User.is_admin == True).scalar() or 0

    cutoff = datetime.utcnow() - timedelta(days=30)
    recent_reg = db.query(func.count(User.id)).filter(User.created_at >= cutoff).scalar() or 0
    active_last_30 = db.query(func.count(User.id)).filter(User.last_login >= cutoff).scalar() or 0

    return create_response(True, data={
        "total_users": int(total_users),
        "active_users": int(active_users),
        "verified_users": int(verified_users),
        "admin_users": int(admin_users),
        "recent_registrations_30d": int(recent_reg),
        "active_last_30d": int(active_last_30),
    })


@router.get("/stats/top-contributors", response_model=Dict[str, Any])
async def top_contributors(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(
            User.id,
            User.username,
            User.full_name,
            func.count(Speech.id).label("speech_count"),
        )
        .join(Speech, Speech.user_id == User.id)
        .filter(Speech.is_public == True)
        .group_by(User.id)
        .order_by(desc("speech_count"))
        .limit(limit)
        .all()
    )

    out = [
        {"user_id": uid, "username": uname, "full_name": fname, "speech_count": int(cnt)}
        for uid, uname, fname, cnt in rows
    ]
    return create_response(True, data={"contributors": out, "count": len(out)})


# =============================================================================
# PUBLIC PROFILE
# =============================================================================

@router.get("/{user_id}/public", response_model=PublicUserProfile)
async def get_public_profile(
    user_id: int,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    speech_count = db.query(func.count(Speech.id)).filter(Speech.user_id == user_id, Speech.is_public == True).scalar() or 0
    analysis_count = (
        db.query(func.count(Analysis.id))
        .join(Speech)
        .filter(Speech.user_id == user_id, Speech.is_public == True)
        .scalar()
        or 0
    )

    profile = PublicUserProfile(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        organization=user.organization,
        bio=user.bio,
        created_at=user.created_at,
        speech_count=int(speech_count),
        analysis_count=int(analysis_count),
    )
    return profile


# =============================================================================
# USER ACTIVITY & STATS (protected)
# =============================================================================

@router.get("/{user_id}/speeches", response_model=Dict[str, Any])
async def get_user_speeches(
    user_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
):
    is_admin = bool(getattr(current_user, "is_admin", False))
    is_self = current_user.id == user_id

    q = db.query(Speech).filter(Speech.user_id == user_id)

    if not is_self and not is_admin:
        q = q.filter(Speech.is_public == True)

    total = q.count()
    offset = (page - 1) * page_size
    rows = q.order_by(desc(Speech.created_at)).offset(offset).limit(page_size).all()

    speeches = [
        {
            "id": s.id,
            "title": s.title,
            "speaker": s.speaker,
            "date": s.date.isoformat() if s.date else None,
            "word_count": s.word_count,
            "status": s.status,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "has_analysis": bool(s.analysis is not None),
        }
        for s in rows
    ]

    return create_response(True, data={
        "speeches": speeches,
        "total": int(total),
        "page": page,
        "page_size": page_size,
    })


@router.get("/{user_id}/stats", response_model=Dict[str, Any])
async def get_user_stats(
    user_id: int,
    current_user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
):
    is_admin = bool(getattr(current_user, "is_admin", False))
    if current_user.id != user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    stats = _get_user_statistics(db, user_id)
    return stats.dict()


@router.get("/{user_id}/activity", response_model=Dict[str, Any])
async def get_user_activity(
    user_id: int,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
):
    is_admin = bool(getattr(current_user, "is_admin", False))
    if current_user.id != user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")

    cutoff = datetime.utcnow() - timedelta(days=days)

    speeches = (
        db.query(Speech)
        .filter(Speech.user_id == user_id, Speech.created_at >= cutoff)
        .order_by(desc(Speech.created_at))
        .limit(15)
        .all()
    )

    activity: List[Dict[str, Any]] = []
    for s in speeches:
        activity.append({
            "type": "speech_created",
            "timestamp": s.created_at.isoformat() if s.created_at else None,
            "data": {"speech_id": s.id, "title": s.title, "speaker": s.speaker},
        })
        if s.analyzed_at and s.analyzed_at >= cutoff:
            activity.append({
                "type": "speech_analyzed",
                "timestamp": s.analyzed_at.isoformat(),
                "data": {"speech_id": s.id, "title": s.title, "ideology": s.analysis.ideology_family if s.analysis else None},
            })

    activity.sort(key=lambda x: (x["timestamp"] or ""), reverse=True)

    return create_response(True, data={
        "activity": activity,
        "period_days": int(days),
        "activity_count": len(activity),
    })


# =============================================================================
# ADMIN USER MANAGEMENT
# =============================================================================

@router.put("/{user_id}/activate", response_model=Dict[str, Any])
async def activate_user(
    user_id: int,
    activate: bool = Query(...),
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id and not activate:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    user.is_active = bool(activate)
    user.updated_at = datetime.utcnow()
    db.commit()

    return create_response(True, message=f"User {'activated' if activate else 'deactivated'} successfully")


@router.put("/{user_id}/verify", response_model=Dict[str, Any])
async def verify_user(
    user_id: int,
    verify: bool = Query(...),
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_verified = bool(verify)
    user.updated_at = datetime.utcnow()
    db.commit()

    return create_response(True, message=f"User {'verified' if verify else 'unverified'} successfully")


@router.put("/{user_id}/admin", response_model=Dict[str, Any])
async def toggle_admin(
    user_id: int,
    grant: bool = Query(...),
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id and not grant:
        raise HTTPException(status_code=400, detail="Cannot revoke your own admin privileges")

    user.is_admin = bool(grant)
    user.updated_at = datetime.utcnow()
    db.commit()

    return create_response(True, message=f"Admin privileges {'granted' if grant else 'revoked'} successfully")


@router.put("/{user_id}", response_model=Dict[str, Any])
async def update_user_admin(
    user_id: int,
    payload: UserUpdateAdmin,
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update = payload.dict(exclude_unset=True)

    if "email" in update and update["email"] != user.email:
        exists = db.query(User).filter(User.email == update["email"]).first()
        if exists:
            raise HTTPException(status_code=400, detail="Email already in use")

    if "username" in update and update["username"] != user.username:
        exists = db.query(User).filter(User.username == update["username"]).first()
        if exists:
            raise HTTPException(status_code=400, detail="Username already taken")

    if user.id == admin.id and "is_admin" in update and update["is_admin"] is False:
        raise HTTPException(status_code=400, detail="Cannot remove your own admin privileges")

    for k, v in update.items():
        setattr(user, k, v)

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return create_response(True, message="User updated successfully", data={
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "is_admin": user.is_admin,
    })


@router.delete("/{user_id}", response_model=Dict[str, Any])
async def delete_user_admin(
    user_id: int,
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    email = user.email
    db.delete(user)
    db.commit()

    return create_response(True, message=f"User {email} deleted successfully")


# =============================================================================
# BULK OPERATIONS (admin)
# =============================================================================

@router.post("/bulk-update", response_model=Dict[str, Any])
async def bulk_update_users(
    payload: BulkUpdateRequest,
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    users = db.query(User).filter(User.id.in_(payload.user_ids)).all()
    if not users:
        raise HTTPException(status_code=404, detail="No users found for given IDs")

    if admin.id in payload.user_ids:
        if payload.updates.get("is_active") is False:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account in bulk update")
        if payload.updates.get("is_admin") is False:
            raise HTTPException(status_code=400, detail="Cannot remove your own admin in bulk update")

    allowed_fields = {"is_active", "is_verified", "is_admin", "organization", "bio", "full_name"}
    updates = {k: v for k, v in (payload.updates or {}).items() if k in allowed_fields}

    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    for u in users:
        for k, v in updates.items():
            setattr(u, k, v)
        u.updated_at = datetime.utcnow()

    db.commit()
    return create_response(True, message=f"Updated {len(users)} users", data={"updated_count": len(users)})


@router.post("/bulk-delete", response_model=Dict[str, Any])
async def bulk_delete_users(
    payload: BulkDeleteRequest,
    admin: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Must set confirm=true to delete users")

    if admin.id in payload.user_ids:
        raise HTTPException(status_code=400, detail="Cannot delete your own account in bulk delete")

    users = db.query(User).filter(User.id.in_(payload.user_ids)).all()
    if not users:
        raise HTTPException(status_code=404, detail="No users found for given IDs")

    deleted_emails = [u.email for u in users]
    for u in users:
        db.delete(u)

    db.commit()
    return create_response(True, message=f"Deleted {len(users)} users", data={"deleted_count": len(users), "deleted_emails": deleted_emails})


__all__ = ["router"]