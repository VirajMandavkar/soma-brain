"""
SOMA Brain — T-04: MRI Report Parameter Extractor

Takes a clinical MRI report (PDF or text) and uses Gemma 4 E4B (via Ollama)
to extract structured PatientParams.

Why this matters: This is the bridge between clinical radiology and
computational neuroscience. No existing tool automates this — radiologists
produce free-text reports, but TVB needs numerical parameters. Gemma 4
reads the report like a neuroradiologist and outputs structured JSON.

Pipeline:
  1. PDF → raw text (PyMuPDF)
  2. Raw text → Gemma 4 prompt with PatientParams JSON schema
  3. Gemma 4 response → parse JSON → validate with Pydantic
  4. Fill missing fields with age/sex-adjusted population defaults
"""

import json
import re
from pathlib import Path

import httpx
from loguru import logger

from soma.config import get_settings
from soma.patient.schemas import PatientParams

# Population defaults for missing MRI values — derived from ADNI normative data.
# These are age-band midpoints for healthy elderly (65-75).
# When a value can't be extracted from the report, we use these and flag it
# in missing_fields so downstream modules know the uncertainty is higher.
POPULATION_DEFAULTS = {
    "hippocampal_volume_normalized": 0.85,
    "entorhinal_cortex_volume_normalized": 0.88,
    "prefrontal_volume_normalized": 0.90,
    "whole_brain_volume_normalized": 0.92,
    "global_fa_score": 0.45,
    "hippocampal_cingulum_fa": 0.40,
    "connectivity_scale_factor": 0.85,
    "hippocampal_pfc_connection_strength": 0.70,
    "disease_state": "MCI",
    "estimated_amyloid_burden": 0.3,
    "patient_age": 72,
    "patient_sex": "unknown",
}

# The extraction prompt. This is the most critical prompt in the system —
# it determines how accurately we parameterize the brain twin.
# Key design choices:
#   - We provide the exact JSON schema so Gemma 4 knows the output format
#   - We explain what each field means in clinical terms (not just field names)
#   - We ask for extraction_confidence so we know when to trust less
#   - We explicitly say "use null if not found" to avoid hallucinated values
EXTRACTION_PROMPT = """You are an expert neuroradiologist extracting structured parameters from an MRI report for computational brain modeling.

Read the following MRI report carefully and extract ALL available parameters into the JSON schema below. For each parameter:
- If the value is explicitly stated in the report, extract it
- If the value can be reasonably inferred from the report findings, infer it and note lower confidence
- If the value cannot be determined, set it to null

IMPORTANT CLINICAL MAPPINGS:
- hippocampal_volume_normalized: 1.0 = age-matched healthy mean. "Mild atrophy" ~ 0.75-0.85, "Moderate atrophy" ~ 0.55-0.75, "Severe atrophy" ~ 0.35-0.55
- entorhinal_cortex_volume_normalized: Same scale. Often atrophied before hippocampus in AD.
- prefrontal_volume_normalized: Same scale. "Age-appropriate" = 0.85-1.0
- whole_brain_volume_normalized: Same scale. "Mild global atrophy" ~ 0.80-0.90
- global_fa_score: Fractional anisotropy. Normal elderly = 0.40-0.50. "Reduced white matter integrity" ~ 0.30-0.40
- hippocampal_cingulum_fa: Key Alzheimer's tract. Normal = 0.38-0.48. "Disrupted" ~ 0.25-0.35
- connectivity_scale_factor: 1.0 = fully connected healthy brain. Reduce based on white matter disease burden.
- hippocampal_pfc_connection_strength: 1.0 = healthy. Reduce with reported disconnection.
- disease_state: One of "healthy", "MCI", "early_AD", "moderate_AD". Match to clinical impression.
- estimated_amyloid_burden: 0-1 scale. If PET data available, use it. Otherwise estimate from atrophy pattern.
- extraction_confidence: 0-1. How confident are you in this extraction overall?

OUTPUT FORMAT — return ONLY valid JSON, no other text:
{
    "hippocampal_volume_normalized": <float 0-1 or null>,
    "entorhinal_cortex_volume_normalized": <float 0-1 or null>,
    "prefrontal_volume_normalized": <float 0-1 or null>,
    "whole_brain_volume_normalized": <float 0-1 or null>,
    "global_fa_score": <float 0-1 or null>,
    "hippocampal_cingulum_fa": <float 0-1 or null>,
    "connectivity_scale_factor": <float 0.5-1.0 or null>,
    "hippocampal_pfc_connection_strength": <float 0-1 or null>,
    "disease_state": <"healthy"|"MCI"|"early_AD"|"moderate_AD">,
    "estimated_amyloid_burden": <float 0-1 or null>,
    "patient_age": <int or null>,
    "patient_sex": <"M"|"F"|"unknown">,
    "extraction_confidence": <float 0-1>
}

MRI REPORT:
---
{report_text}
---

Extract the parameters as JSON:"""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF.

    PyMuPDF (fitz) is used instead of pdfplumber or PyPDF2 because it handles
    the widest variety of PDF encodings and scanned-text layouts that clinical
    reports often use.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    logger.info("Extracted {} chars from {} pages of {}", len(text), len(pages), pdf_path)
    return text


def _call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Send a prompt to Google Gemini API and return the response text.

    Uses the google-genai SDK. Free tier has rate limits (~15 RPM, ~1M tokens/day).
    Retries on 429 (rate limit) with exponential backoff.
    """
    import time as _time
    from google import genai
    from google.genai import errors as genai_errors

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                },
            )
            return response.text
        except genai_errors.ClientError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1) * 15  # 30s, 60s, 120s
                logger.warning("Gemini rate limit hit, retrying in {}s...", wait)
                _time.sleep(wait)
            else:
                raise


def _call_ollama(prompt: str, model: str | None = None) -> str:
    """Send a prompt to Ollama (local) and return the response text.

    Fallback for systems with GPU that can run local models.
    """
    settings = get_settings()
    model = model or settings.gemma_model

    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
            },
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"]


def _call_llm(prompt: str, model: str | None = None) -> str:
    """Route to Gemini API (preferred) or Ollama (fallback).

    Gemini API works on any system without GPU. Ollama requires local
    model installation. We check which is configured and use that.
    """
    settings = get_settings()

    if settings.use_gemini:
        logger.info("Using Gemini API ({})", settings.gemini_model)
        return _call_gemini(prompt)
    else:
        logger.info("Using Ollama local ({})", settings.gemma_model)
        return _call_ollama(prompt, model)


def _parse_json_from_response(response_text: str) -> dict:
    """Extract JSON from Gemma's response, handling common formatting issues.

    LLMs sometimes wrap JSON in markdown code blocks or add explanatory text
    before/after. This function handles those cases robustly.
    """
    # Try direct parse first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response_text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response: {response_text[:200]}...")


def _fill_defaults(extracted: dict) -> tuple[dict, list[str]]:
    """Replace null/missing fields with population defaults. Returns filled dict and missing field names.

    This is critical for robustness — a real MRI report might not mention
    entorhinal cortex volume, but TVB still needs a value. We use conservative
    population defaults and flag the field so Monte Carlo samples it with
    wider uncertainty.
    """
    missing = []
    filled = {}

    for field_name, default_value in POPULATION_DEFAULTS.items():
        value = extracted.get(field_name)
        if value is None:
            filled[field_name] = default_value
            missing.append(field_name)
        else:
            filled[field_name] = value

    # Preserve extraction_confidence from model
    filled["extraction_confidence"] = extracted.get("extraction_confidence", 0.5)
    filled["missing_fields"] = missing

    return filled, missing


def extract_patient_params(
    source: str,
    model: str | None = None,
) -> PatientParams:
    """Extract PatientParams from an MRI report PDF or raw text.

    Args:
        source: Path to a PDF file, or raw report text string.
        model: Ollama model name override (default: from settings).

    Returns:
        Validated PatientParams with missing fields filled from population defaults.

    This is the main entry point for T-04. It chains together:
        PDF parsing → prompt construction → Ollama call → JSON parsing → default filling → Pydantic validation
    """
    # Step 1: Get raw text
    source_path = Path(source)
    if source_path.exists() and source_path.suffix.lower() == ".pdf":
        report_text = extract_text_from_pdf(source)
    else:
        report_text = source  # assume raw text

    if len(report_text) < 50:
        raise ValueError(f"Report text too short ({len(report_text)} chars) — likely extraction failure")

    logger.info("Extracting patient params from report ({} chars)", len(report_text))

    # Step 2: Build prompt and call LLM (Gemini API or Ollama)
    prompt = EXTRACTION_PROMPT.format(report_text=report_text)
    raw_response = _call_llm(prompt, model=model)
    logger.debug("LLM response: {}", raw_response[:500])

    # Step 3: Parse JSON
    extracted = _parse_json_from_response(raw_response)

    # Step 4: Fill defaults for missing fields
    filled, missing = _fill_defaults(extracted)
    if missing:
        logger.warning("Missing fields filled with defaults: {}", missing)

    # Step 5: Validate with Pydantic
    params = PatientParams(**filled)
    logger.info(
        "Extracted: disease_state={}, confidence={:.2f}, missing={}",
        params.disease_state,
        params.extraction_confidence,
        params.missing_fields,
    )

    return params


# --- Synthetic test reports for validation ---

SAMPLE_REPORT_MODERATE_AD = """
BRAIN MRI WITH AND WITHOUT CONTRAST

PATIENT: Jane Doe, 74F
DATE: 2026-03-15
INDICATION: Progressive memory loss, 2-year history. Rule out structural etiology.

TECHNIQUE: 3T MRI. Sagittal T1 MPRAGE, axial T2, FLAIR, DWI, SWI, DTI (30 directions).

FINDINGS:

GRAY MATTER:
Bilateral hippocampal atrophy, moderate severity (volume approximately 60% of age-matched
normal). The entorhinal cortex shows moderate volume loss. Temporal lobe atrophy is
predominant over frontal regions. Prefrontal volume appears mildly reduced, approximately
85% of expected. Global brain volume is reduced, consistent with moderate generalized atrophy.

WHITE MATTER:
Moderate periventricular and deep white matter hyperintensities on FLAIR, Fazekas grade 2.
DTI shows reduced fractional anisotropy in the hippocampal cingulum bundle (FA = 0.31)
and globally (mean FA = 0.37). Tractography demonstrates reduced hippocampal-prefrontal
connectivity compared to normative atlas.

OTHER: No acute infarction. No mass lesion. Mild chronic microhemorrhages on SWI (2 foci,
parietal). Ventricles are moderately enlarged consistent with ex vacuo hydrocephalus.

IMPRESSION:
Findings are consistent with moderate Alzheimer's disease. Bilateral hippocampal and
entorhinal cortex atrophy with associated white matter tract disruption. Recommend
correlation with amyloid PET and neuropsychological testing. Estimated Braak stage IV-V
based on atrophy pattern.
"""

SAMPLE_REPORT_MCI = """
BRAIN MRI WITHOUT CONTRAST

PATIENT: John Smith, 68M
DATE: 2026-02-20
INDICATION: Subjective memory complaints. Family history of Alzheimer's.

TECHNIQUE: 1.5T MRI. Standard sequences including T1 volumetric, FLAIR, DWI.

FINDINGS:

GRAY MATTER:
Mild bilateral hippocampal volume reduction. Hippocampal volume estimated at approximately
78% of age-matched normative values. Entorhinal cortex volume within low-normal range.
No significant frontal or parietal atrophy. Whole brain volume is age-appropriate.

WHITE MATTER:
Minimal scattered periventricular white matter hyperintensities, Fazekas grade 1.
No DTI was performed.

OTHER: No acute pathology. Normal ventricle size.

IMPRESSION:
Mild hippocampal volume loss that may represent early neurodegenerative change.
Clinical correlation recommended. Consider follow-up MRI in 12 months to assess
interval change. No evidence of vascular dementia or other structural pathology.
"""

SAMPLE_REPORT_HEALTHY = """
BRAIN MRI WITHOUT CONTRAST

PATIENT: Robert Chen, 70M
DATE: 2026-01-10
INDICATION: Screening. No cognitive complaints.

TECHNIQUE: 3T MRI. T1 MPRAGE, FLAIR, DWI, DTI (64 directions).

FINDINGS:

GRAY MATTER:
Normal hippocampal volume bilaterally. No focal cortical atrophy. Brain parenchymal
volume is age-appropriate with no significant generalized atrophy.

WHITE MATTER:
Minimal age-related white matter changes (Fazekas 1). DTI shows normal fractional
anisotropy globally (mean FA = 0.46) and in the hippocampal cingulum (FA = 0.44).
Normal-appearing white matter tracts on tractography.

OTHER: No acute pathology. Normal ventricles and sulci for age.

IMPRESSION:
Normal brain MRI for age. No evidence of neurodegenerative disease or significant
vascular pathology.
"""


def run_extraction_tests():
    """Test extraction on 3 synthetic reports. Requires Ollama running with gemma3:4b."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]SOMA Brain -- MRI Extraction Test (T-04)[/bold cyan]\n")

    test_cases = [
        ("Moderate AD (74F)", SAMPLE_REPORT_MODERATE_AD, "moderate_AD"),
        ("MCI (68M)", SAMPLE_REPORT_MCI, "MCI"),
        ("Healthy (70M)", SAMPLE_REPORT_HEALTHY, "healthy"),
    ]

    all_passed = True
    for name, report_text, expected_state in test_cases:
        console.print(f"  Testing: {name}...", end=" ")
        try:
            params = extract_patient_params(report_text)
            state_match = params.disease_state == expected_state
            status = "[green]OK[/green]" if state_match else "[yellow]MISMATCH[/yellow]"
            console.print(
                f"{status} state={params.disease_state} (expected {expected_state}), "
                f"hippo={params.hippocampal_volume_normalized:.2f}, "
                f"conf={params.extraction_confidence:.2f}, "
                f"missing={len(params.missing_fields)} fields"
            )
            if not state_match:
                all_passed = False
        except Exception as e:
            console.print(f"[red]ERROR[/red] {e}")
            all_passed = False

    if all_passed:
        console.print("\n[bold green]All extraction tests passed![/bold green]")
    else:
        console.print("\n[yellow]Some tests had mismatches -- review output above.[/yellow]")

    return all_passed


if __name__ == "__main__":
    run_extraction_tests()
