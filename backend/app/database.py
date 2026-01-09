"""
Database configuration and session management.
Uses SQLite for MVP - easily upgradeable to PostgreSQL.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# SQLite database file path
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dca_management.db")

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Required for SQLite
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from . import models  # Import models to register them
    Base.metadata.create_all(bind=engine)
