"""Database models for job persistence."""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./nexusmd.db")

# psycopg2 requires postgresql:// scheme; Railway provides postgres:// — normalise it
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    # SQLite needs check_same_thread=False; ignored by PostgreSQL
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DockingJob(Base):
    """Persistent docking job record."""
    __tablename__ = "docking_jobs"

    job_id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Input
    protein_id = Column(String)
    ligand_count = Column(Integer)

    # Results
    completed_ligands = Column(Integer, default=0)
    poses_generated = Column(Integer, default=0)

    # Storage
    results_bucket_path = Column(String, nullable=True)  # s3://bucket/job_id/
    results_sdf_url = Column(String, nullable=True)
    results_zip_url = Column(String, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Metadata
    job_metadata = Column(JSON, nullable=True)  # Extra data as JSON


def init_db():
    """Create all tables. Called at application startup."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        import logging
        logging.getLogger("nexusmd.database").warning(
            f"Database table creation failed (continuing without persistence): {e}"
        )


def get_db():
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
