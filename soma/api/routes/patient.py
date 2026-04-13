"""SOMA Brain — Patient Twin API Routes"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ParameterizeRequest(BaseModel):
    mri_report_base64: str
    patient_id: str


@router.post("/parameterize")
async def parameterize_patient(request: ParameterizeRequest):
    """Extract patient parameters from MRI report and build brain digital twin."""
    # Will be implemented in T-04 (MRI extractor) + T-05 (twin builder)
    return {
        "twin_id": "placeholder-will-be-uuid",
        "extraction_confidence": 0.0,
        "message": "Not yet implemented — see T-04",
    }
