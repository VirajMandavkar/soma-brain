"""
SOMA Brain — Simulation API Routes

POST /submit  → queue a new drug simulation
GET /{job_id}/status → poll current progress
GET /{job_id}/report → get completed results
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SimulationRequest(BaseModel):
    compound_smiles: str
    patient_twin_id: UUID
    priority: str = "standard"


class SimulationStatusResponse(BaseModel):
    job_id: UUID
    status: str
    progress_pct: float = 0.0
    current_stage: str = "queued"
    elapsed_seconds: float = 0.0


@router.post("/submit")
async def submit_simulation(request: SimulationRequest):
    """Submit a drug candidate for full SOMA pipeline evaluation."""
    # Will be wired to Celery worker in T-18
    return {
        "job_id": "placeholder-will-be-uuid",
        "status": "queued",
        "estimated_minutes": 20,
    }


@router.get("/{job_id}/status")
async def get_simulation_status(job_id: UUID):
    """Poll the current status of a running simulation."""
    # Will query PostgreSQL simulation_jobs table in T-19
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.get("/{job_id}/report")
async def get_simulation_report(job_id: UUID):
    """Get the full report for a completed simulation."""
    # Will return MonteCarloResult + reasoning chains in T-19
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
