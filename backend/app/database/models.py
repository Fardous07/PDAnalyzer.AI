from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
    false,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

JSONDictType = MutableDict.as_mutable(JSON().with_variant(JSONB, "postgresql"))
JSONListType = MutableList.as_mutable(JSON().with_variant(JSONB, "postgresql"))

speech_project_association = Table(
    "speech_project_association",
    Base.metadata,
    Column("speech_id", Integer, ForeignKey("speeches.id", ondelete="CASCADE"), primary_key=True),
    Column("project_id", Integer, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("added_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("sort_order", Integer, default=0, nullable=False),
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)

    full_name = Column(String(200), nullable=True)
    organization = Column(String(200), nullable=True)
    bio = Column(Text, nullable=True)

    preferences = Column(JSONDictType, nullable=False, default=dict)

    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    is_admin = Column(Boolean, default=False, nullable=False)
    role = Column(String(50), default="user", nullable=False)

    subscription_tier = Column(String(50), default="free", nullable=False)
    max_speeches = Column(Integer, default=50, nullable=False)
    max_file_size = Column(Integer, default=100_000_000, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

    speeches = relationship("Speech", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


class Speech(Base):
    __tablename__ = "speeches"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    title = Column(String(500), nullable=False, index=True)
    speaker = Column(String(200), nullable=False, index=True)
    date = Column(DateTime, nullable=True, index=True)
    location = Column(String(200), nullable=True)
    event = Column(String(200), nullable=True)

    text = Column(Text, nullable=False)
    language = Column(String(10), default="en", nullable=False)
    word_count = Column(Integer, nullable=True)

    source_url = Column(String(500), nullable=True)
    source_type = Column(String(50), nullable=True)
    media_url = Column(String(500), nullable=True)

    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)

    party = Column(String(100), nullable=True)
    role = Column(String(100), nullable=True)

    use_semantic_segmentation = Column(Boolean, default=True, nullable=False, server_default=true())
    use_semantic_scoring = Column(Boolean, default=True, nullable=False, server_default=true())
    use_research_analysis = Column(Boolean, default=False, nullable=False, server_default=false())

    status = Column(String(20), default="pending", nullable=False)
    is_public = Column(Boolean, default=False, nullable=False, server_default=false())

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    analyzed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="speeches")
    analysis = relationship("Analysis", back_populates="speech", uselist=False, cascade="all, delete-orphan")
    questions = relationship("Question", back_populates="speech", cascade="all, delete-orphan")
    projects = relationship("Project", secondary=speech_project_association, back_populates="speeches")

    __table_args__ = (
        Index("idx_speeches_user_created", "user_id", "created_at"),
        Index("idx_speeches_speaker_date", "speaker", "date"),
    )

    def __repr__(self) -> str:
        return f"<Speech(id={self.id}, title='{self.title}', speaker='{self.speaker}')>"


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    speech_id = Column(
        Integer,
        ForeignKey("speeches.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    ideology_family = Column(String(50), nullable=False, index=True)
    ideology_subtype = Column(String(100), nullable=True, index=True)

    libertarian_score = Column(Float, nullable=False, default=0.0)
    authoritarian_score = Column(Float, nullable=False, default=0.0)
    centrist_score = Column(Float, nullable=False, default=0.0)

    confidence_score = Column(Float, nullable=False, default=0.0)

    marpor_codes = Column(JSONListType, nullable=False, default=list)
    full_results = Column(JSONDictType, nullable=False, default=dict)

    processing_time_seconds = Column(Float, nullable=True)
    segment_count = Column(Integer, nullable=True)
    siu_count = Column(Integer, nullable=True)
    key_statement_count = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    speech = relationship("Speech", back_populates="analysis")
    history = relationship("AnalysisHistory", back_populates="analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_analysis_ideology", "ideology_family", "ideology_subtype"),
        Index("idx_analysis_scores", "libertarian_score", "authoritarian_score", "centrist_score"),
        Index("idx_analysis_confidence", "confidence_score"),
        CheckConstraint("libertarian_score >= 0 AND libertarian_score <= 100", name="check_lib_score"),
        CheckConstraint("authoritarian_score >= 0 AND authoritarian_score <= 100", name="check_auth_score"),
        CheckConstraint("centrist_score >= 0 AND centrist_score <= 100", name="check_centrist_score"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence"),
        CheckConstraint("(ideology_family <> 'Centrist') OR (ideology_subtype IS NULL)", name="check_centrist_no_subtype"),
    )

    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, speech_id={self.speech_id}, ideology={self.ideology_family})>"

    def get_speech_level(self) -> Dict[str, Any]:
        return (self.full_results or {}).get("speech_level", {}) if isinstance(self.full_results, dict) else {}

    def get_marpor_breakdown(self) -> Dict[str, Any]:
        sl = self.get_speech_level()
        return sl.get("marpor_breakdown", {}) if isinstance(sl, dict) else {}

    def get_key_statements(self) -> List[Dict[str, Any]]:
        fr = self.full_results if isinstance(self.full_results, dict) else {}
        ks = fr.get("key_statements", [])
        return ks if isinstance(ks, list) else []

    def get_sections(self) -> List[Dict[str, Any]]:
        fr = self.full_results if isinstance(self.full_results, dict) else {}
        secs = fr.get("sections", [])
        return secs if isinstance(secs, list) else []


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)

    speech_id = Column(Integer, ForeignKey("speeches.id", ondelete="CASCADE"), nullable=False, index=True)

    question_text = Column(Text, nullable=False)
    question_type = Column(String(20), nullable=False)
    question_order = Column(Integer, default=0, nullable=False)

    generated_by_llm = Column(String(100), nullable=True)
    generation_prompt = Column(Text, nullable=True)

    is_answered = Column(Boolean, default=False, nullable=False)
    answer_text = Column(Text, nullable=True)
    answered_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    speech = relationship("Speech", back_populates="questions")

    __table_args__ = (
        Index("idx_questions_speech_type", "speech_id", "question_type"),
    )

    def __repr__(self) -> str:
        return f"<Question(id={self.id}, speech_id={self.speech_id}, type='{self.question_type}')>"


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    is_public = Column(Boolean, default=False, nullable=False)
    color = Column(String(20), nullable=True)

    speech_count = Column(Integer, default=0, nullable=False)
    avg_libertarian_score = Column(Float, nullable=True)
    avg_authoritarian_score = Column(Float, nullable=True)
    avg_centrist_score = Column(Float, nullable=True)
    dominant_ideology = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="projects")
    speeches = relationship("Speech", secondary=speech_project_association, back_populates="projects")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="unique_user_project_name"),
        Index("idx_projects_user_updated", "user_id", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}', speech_count={self.speech_count})>"


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)

    ideology_family = Column(String(50), nullable=False)
    ideology_subtype = Column(String(100), nullable=True)

    libertarian_score = Column(Float, nullable=False, default=0.0)
    authoritarian_score = Column(Float, nullable=False, default=0.0)
    centrist_score = Column(Float, nullable=False, default=0.0)

    confidence_score = Column(Float, nullable=False, default=0.0)

    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)
    use_semantic = Column(Boolean, default=True, nullable=False)

    change_reason = Column(String(200), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    analysis = relationship("Analysis", back_populates="history")

    __table_args__ = (
        CheckConstraint("libertarian_score >= 0 AND libertarian_score <= 100", name="check_history_lib_score"),
        CheckConstraint("authoritarian_score >= 0 AND authoritarian_score <= 100", name="check_history_auth_score"),
        CheckConstraint("centrist_score >= 0 AND centrist_score <= 100", name="check_history_centrist_score"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_history_confidence"),
        CheckConstraint("(ideology_family <> 'Centrist') OR (ideology_subtype IS NULL)", name="check_history_centrist_no_subtype"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisHistory(id={self.id}, analysis_id={self.analysis_id}, date={self.created_at})>"


def create_all_tables(engine) -> None:
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine) -> None:
    Base.metadata.drop_all(bind=engine)


def get_table_names() -> List[str]:
    return [t.name for t in Base.metadata.sorted_tables]


__all__ = [
    "Base",
    "User",
    "Speech",
    "Analysis",
    "Question",
    "Project",
    "AnalysisHistory",
    "speech_project_association",
    "create_all_tables",
    "drop_all_tables",
    "get_table_names",
]