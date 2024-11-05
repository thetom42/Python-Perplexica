from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from db.models import Base

# Create SQLite database engine
engine = create_engine('sqlite:///data/db.sqlite')
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Get a database session - similar to how Drizzle's db instance is used
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Export session factory for use in routes
db = SessionLocal
