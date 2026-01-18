"""
AUTH MODELS - Authentication and User Models
=============================================

LOCATION: backend/app/models/auth_models.py

This module contains authentication-related models and utilities.
Separated from main models.py for better organization.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableDict

from app.database.base import Base


class User(Base):
    """
    User model for authentication and profiles.
    
    Stores user information, authentication data, and preferences.
    """
    __tablename__ = "users"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)
    
    # Profile
    full_name = Column(String(200), nullable=True)
    organization = Column(String(200), nullable=True)
    bio = Column(String(500), nullable=True)
    
    # Preferences (stored as JSON)
    preferences = Column(
        MutableDict.as_mutable(JSON),
        nullable=True,
        default=dict,
        comment="User preferences: theme, default_llm, etc."
    )
    
    # User limits (from env or custom)
    max_speeches = Column(Integer, nullable=True, comment="Max speeches allowed (null = unlimited)")
    max_file_size = Column(Integer, nullable=True, comment="Max file size in bytes (null = use default)")
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships (defined in other model files to avoid circular imports)
    # speeches = relationship("Speech", back_populates="user", cascade="all, delete-orphan")
    # projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>"
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert user to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive data (password hash)
        
        Returns:
            dict: User data
        """
        data = {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "organization": self.organization,
            "bio": self.bio,
            "preferences": self.preferences,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }
        
        if include_sensitive:
            data["hashed_password"] = self.hashed_password
        
        return data
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference by key."""
        if self.preferences:
            return self.preferences.get(key, default)
        return default
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference."""
        if self.preferences is None:
            self.preferences = {}
        self.preferences[key] = value


# Export
__all__ = ["User"]