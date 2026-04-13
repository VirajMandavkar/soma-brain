"""
SOMA Brain — Patient Data Models

These Pydantic models define the contract between:
  - MRI extractor (Gemma 4 output) → PatientParams
  - Twin builder (TVB parameterization) → PatientTwin
  - Monte Carlo sampler (uncertainty quantification) → uses PatientParams ranges

Every downstream module validates against these schemas, so a malformed
extraction from Gemma 4 is caught here — not 3 layers deep in TVB.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PatientParams(BaseModel):
    """Structured parameters extracted from a clinical MRI report by Gemma 4."""

    # --- Structural measures (from MRI volumetry) ---
    hippocampal_volume_normalized: float = Field(
        ge=0.0, le=1.0,
        description="Hippocampal volume as fraction of healthy population mean (0-1)",
    )
    entorhinal_cortex_volume_normalized: float = Field(
        ge=0.0, le=1.0,
        description="Entorhinal cortex volume as fraction of healthy mean",
    )
    prefrontal_volume_normalized: float = Field(
        ge=0.0, le=1.0,
        description="Prefrontal cortex volume as fraction of healthy mean",
    )
    whole_brain_volume_normalized: float = Field(
        ge=0.0, le=1.0,
        description="Whole brain volume as fraction of healthy mean",
    )

    # --- White matter integrity (from DTI) ---
    global_fa_score: float = Field(
        ge=0.0, le=1.0,
        description="Global fractional anisotropy (0=no directionality, 1=perfect tracts)",
    )
    hippocampal_cingulum_fa: float = Field(
        ge=0.0, le=1.0,
        description="FA of the hippocampal cingulum bundle — key Alzheimer's pathway",
    )

    # --- Connectivity (derived from DTI tractography) ---
    connectivity_scale_factor: float = Field(
        ge=0.5, le=1.0,
        description="Global connectivity scaling vs healthy reference (1.0 = healthy)",
    )
    hippocampal_pfc_connection_strength: float = Field(
        ge=0.0, le=1.0,
        description="Hippocampus-to-prefrontal connection strength (normalized)",
    )

    # --- Disease state markers ---
    disease_state: Literal["healthy", "MCI", "early_AD", "moderate_AD"] = Field(
        description="Clinical classification from radiology impression",
    )
    estimated_amyloid_burden: float = Field(
        ge=0.0, le=1.0,
        description="Estimated amyloid load (from PET if available, else inferred)",
    )

    # --- Demographics ---
    patient_age: int = Field(ge=18, le=120)
    patient_sex: Literal["M", "F", "unknown"] = Field(default="unknown")

    # --- Extraction metadata ---
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Gemma 4's self-reported confidence in the extraction",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Parameter names not found in the report (filled with population defaults)",
    )


class PatientTwin(BaseModel):
    """A configured brain digital twin ready for simulation."""

    twin_id: UUID = Field(default_factory=uuid4)
    patient_params: PatientParams
    connectivity_checksum: str = Field(
        description="SHA256 of the TVB Connectivity weights matrix — for reproducibility",
    )
    disease_preset: str = Field(
        description="Which TVB parameter preset was applied (e.g., 'moderate_AD_v1')",
    )
    baseline_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Pre-drug simulation metrics: theta_power, gamma_power, synchrony, etc.",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MonteCarloResult(BaseModel):
    """Aggregated output from 20,000 Monte Carlo simulation runs."""

    compound_smiles: str
    compound_name: str
    twin_id: UUID
    n_simulations: int
    n_failed: int = 0

    # --- Primary outcome: theta power improvement ---
    theta_improvement_mean: float
    theta_improvement_median: float
    theta_improvement_std: float
    theta_ci_95_lower: float
    theta_ci_95_upper: float

    # --- Gamma power improvement ---
    gamma_improvement_mean: float
    gamma_ci_95_lower: float
    gamma_ci_95_upper: float

    # --- Synchrony ---
    synchrony_mean: float
    synchrony_ci_95_lower: float
    synchrony_ci_95_upper: float

    # --- Probability metrics ---
    prob_any_improvement: float = Field(
        ge=0.0, le=1.0,
        description="P(theta improvement > 0%)",
    )
    prob_clinically_meaningful: float = Field(
        ge=0.0, le=1.0,
        description="P(theta improvement > 10%)",
    )
    responder_probability: float = Field(
        ge=0.0, le=1.0,
        description="Estimated fraction of patients who would respond",
    )

    # --- Sensitivity analysis ---
    dominant_uncertainty_source: str = Field(
        description="Parameter with highest Sobol S1 index — drives most outcome variance",
    )
    sobol_indices: dict[str, float] = Field(
        default_factory=dict,
        description="Sobol first-order sensitivity indices for each input parameter",
    )
