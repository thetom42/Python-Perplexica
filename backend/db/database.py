"""
Database configuration module for the Perplexica backend.

This module sets up the SQLAlchemy engine, session, and base class for the ORM.
It also provides a function to get a database session.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator

SQLALCHEMY_DATABASE_URL = "sqlite:///./perplexica.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """
    Create and yield a database session.

    This function creates a new SQLAlchemy SessionLocal that will be used in a single request,
    and then closed once the request is finished.

    Yields:
        Generator: A SQLAlchemy SessionLocal object.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
