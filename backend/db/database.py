"""
Database configuration module for the Perplexica backend.

This module sets up the SQLAlchemy engine, session, and base class for the ORM.
It also provides a function to get a database session.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from db.models import Base

# Get database URL from environment variable with fallback
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/db.sqlite')

# Configure connection pool with SQLite specific settings
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create all tables if they don't exist
try:
    Base.metadata.create_all(engine)
except SQLAlchemyError as e:
    raise RuntimeError("Failed to initialize database") from e

# Create session factory with transaction isolation
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

def get_db() -> Generator[Session, None, None]:
    """
    Create and yield a database session.

    This function creates a new SQLAlchemy SessionLocal that will be used in a single request,
    and then closed once the request is finished.

    Yields:
        Session: A SQLAlchemy Session object.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()
