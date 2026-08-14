"""
Database configuration and session management.
Uses SQLite for MVP - easily upgradeable to PostgreSQL.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import logging
from sqlalchemy.engine import make_url

# Load environment variables from .env file
load_dotenv()

# SQLite database file path (default fallback)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dca_management.db")

# Connection arguments
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
else:
    # For Postgres/Supabase connections ensure SSL mode is required when not
    # explicitly provided. Some managed hosts require SSL and will refuse
    # non-SSL connections.
    if DATABASE_URL.startswith("postgres") and "sslmode" not in DATABASE_URL:
        connect_args["sslmode"] = "require"

# Logging: avoid printing credentials
logger = logging.getLogger(__name__)
try:
    parsed = make_url(DATABASE_URL)
    logger.info(f"Database configured: {parsed.drivername}://{parsed.host}:{parsed.port}/{parsed.database}")
except Exception:
    logger.info("Database configured: [unable to parse DATABASE_URL]")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args
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
