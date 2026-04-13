"""
SOMA Brain — PostgreSQL ORM Models (SQLAlchemy 2.0)

Three tables:
  - simulation_jobs: tracks every pipeline run from submission to completion
  - simulation_results: TimescaleDB hypertable storing per-run and aggregate metrics
  - audit_log: regulatory-grade logging of every AI reasoning step

Why TimescaleDB for results: Monte Carlo produces 20,000 rows per drug.
TimescaleDB's columnar compression + time-based partitioning keeps queries
fast even at millions of result rows. Standard PostgreSQL would need manual
partitioning to match this performance.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    BBB_PREDICTION = "bbb_prediction"
    DOCKING = "docking"
    TVB_SIM = "tvb_sim"
    MC_RUNNING = "mc_running"
    REASONING = "reasoning"
    COMPLETE = "complete"
    FAILED = "failed"


class JobPriority(str, enum.Enum):
    HIGH = "high"
    STANDARD = "standard"
    BATCH = "batch"


class SimulationJob(Base):
    __tablename__ = "simulation_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    compound_smiles = Column(String(2048), nullable=False, index=True)
    compound_name = Column(String(512), nullable=True)
    patient_twin_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    priority = Column(Enum(JobPriority), nullable=False, default=JobPriority.STANDARD)
    n_mc_runs = Column(Integer, nullable=False, default=20000)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    celery_task_id = Column(String(256), nullable=True)
    created_by = Column(UUID(as_uuid=True), nullable=True)

    results = relationship("SimulationResult", back_populates="job", lazy="dynamic")
    audit_logs = relationship("AuditLog", back_populates="job", lazy="dynamic")


class SimulationResult(Base):
    """
    TimescaleDB hypertable — partitioned on recorded_at.
    Each row is either a single MC run metric or an aggregate (mc_run_index=None).
    """
    __tablename__ = "simulation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("simulation_jobs.id"), nullable=False, index=True)
    metric_name = Column(String(128), nullable=False)
    value = Column(Float, nullable=False)
    ci_lower = Column(Float, nullable=True)
    ci_upper = Column(Float, nullable=True)
    mc_run_index = Column(Integer, nullable=True)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    job = relationship("SimulationJob", back_populates="results")


class AuditLog(Base):
    """
    Every AI reasoning step is logged here for regulatory traceability.
    If SOMA Brain is ever used in an IND submission, auditors need to
    reconstruct exactly what the AI "thought" at each step.
    """
    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("simulation_jobs.id"), nullable=False, index=True)
    agent_name = Column(String(128), nullable=False)
    input_hash = Column(String(64), nullable=False)
    output_hash = Column(String(64), nullable=False)
    reasoning_chain = Column(JSON, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    job = relationship("SimulationJob", back_populates="audit_logs")
