"""
SOMA Brain -- T-05: TVB Twin Builder

Converts PatientParams (from MRI extraction) into a configured TVB simulation
ready to run. This is the bridge between clinical data and brain simulation.

Pipeline:
  1. PatientParams -> TVB Connectivity (scaled by patient's connectivity)
  2. PatientParams -> JansenRit model parameters (adjusted for disease state)
  3. Run a baseline simulation (no drug) to get reference metrics
  4. Return PatientTwin with connectivity, model config, and baseline metrics

Why patient-specific: A healthy brain and an Alzheimer's brain have fundamentally
different dynamics. Using the same TVB parameters for both would be like testing
aerodynamics on a sports car model and applying the results to a truck.
"""

import hashlib
import time

import numpy as np
from loguru import logger

from soma.patient.schemas import PatientParams, PatientTwin


# Disease state presets -- tuned parameter offsets for each clinical stage.
# These are derived from published TVB Alzheimer's modeling papers:
#   - Zimmermann et al. (2018) NeuroImage: Clinical
#   - Raj et al. (2012) Neuron
# The key insight: Alzheimer's doesn't just reduce connectivity -- it shifts
# the excitatory/inhibitory balance, increases neural noise, and reduces
# long-range coupling preferentially over local coupling.
DISEASE_PRESETS = {
    "healthy": {
        "a_scale": 1.0,        # excitatory gain multiplier
        "b_scale": 1.0,        # inhibitory gain multiplier
        "mu_base": 0.22,       # baseline external drive
        "coupling_a": 0.006,   # global coupling strength
        "noise_nsig": 0.010,   # neural noise level
        "preset_name": "healthy_v1",
    },
    "MCI": {
        "a_scale": 1.05,       # slight excitatory increase (early compensation)
        "b_scale": 0.95,       # slight inhibitory loss
        "mu_base": 0.20,       # reduced drive
        "coupling_a": 0.005,   # slightly reduced coupling
        "noise_nsig": 0.012,   # slightly more noise
        "preset_name": "MCI_v1",
    },
    "early_AD": {
        "a_scale": 1.12,       # excitatory compensation more pronounced
        "b_scale": 0.88,       # more inhibitory loss (GABAergic decline)
        "mu_base": 0.18,       # reduced drive
        "coupling_a": 0.004,   # reduced long-range coupling
        "noise_nsig": 0.015,   # increased noise
        "preset_name": "early_AD_v1",
    },
    "moderate_AD": {
        "a_scale": 1.20,       # strong excitatory shift (hyperexcitability)
        "b_scale": 0.80,       # significant inhibitory loss
        "mu_base": 0.15,       # substantially reduced drive
        "coupling_a": 0.003,   # markedly reduced coupling
        "noise_nsig": 0.020,   # high noise (neuronal loss creates irregularity)
        "preset_name": "moderate_AD_v1",
    },
}


def _load_and_scale_connectivity(params: PatientParams) -> "Connectivity":
    """Load default 76-region connectivity and scale by patient parameters.

    The default connectivity comes from the Human Connectome Project (HCP)
    average. We scale it by the patient's connectivity_scale_factor (from DTI)
    and further attenuate hippocampal connections based on disease severity.
    """
    from tvb.datatypes.connectivity import Connectivity

    conn = Connectivity.from_file()

    # Normalize weights to [0, 1] range (prevents numerical overflow in simulation)
    if conn.weights.max() > 0:
        conn.weights = conn.weights / conn.weights.max()

    # Global scaling by patient connectivity
    conn.weights *= params.connectivity_scale_factor

    # Additional hippocampal attenuation -- hippocampal regions in the 76-region
    # parcellation are typically indices ~34-37 (varies by atlas version).
    # We reduce connections to/from these regions proportional to hippocampal volume loss.
    hippo_indices = list(range(34, 38))  # approximate hippocampal region indices
    hippo_factor = params.hippocampal_volume_normalized
    for idx in hippo_indices:
        if idx < conn.weights.shape[0]:
            conn.weights[idx, :] *= hippo_factor
            conn.weights[:, idx] *= hippo_factor

    # Reduce hippocampal-PFC connections specifically
    pfc_indices = list(range(0, 8))  # approximate prefrontal indices
    pfc_hippo_factor = params.hippocampal_pfc_connection_strength
    for h_idx in hippo_indices:
        for p_idx in pfc_indices:
            if h_idx < conn.weights.shape[0] and p_idx < conn.weights.shape[1]:
                conn.weights[h_idx, p_idx] *= pfc_hippo_factor
                conn.weights[p_idx, h_idx] *= pfc_hippo_factor

    conn.configure()
    return conn


def _build_model(params: PatientParams, preset: dict) -> "JansenRit":
    """Configure JansenRit model parameters based on patient and disease state."""
    from tvb.simulator.models import JansenRit

    # Base excitatory gain scaled by patient hippocampal volume.
    # Rationale: hippocampal volume correlates with preserved excitatory function.
    a_value = 3.25 * preset["a_scale"] * (0.8 + 0.2 * params.hippocampal_volume_normalized)

    # Inhibitory gain -- reduced in AD (GABAergic interneuron loss)
    b_value = 22.0 * preset["b_scale"]

    # External drive -- modulated by overall brain volume
    mu_value = preset["mu_base"] * params.whole_brain_volume_normalized

    model = JansenRit(
        a=np.array([a_value]),
        b=np.array([b_value]),
        mu=np.array([mu_value]),
    )

    return model


def _run_baseline_simulation(conn, model, preset: dict) -> dict:
    """Run a 3-second baseline simulation (no drug) and compute reference metrics.

    These baseline metrics are what we compare drug effects against.
    A drug that increases theta power by 10% relative to baseline is meaningfully
    different from one that doesn't change it.
    """
    from tvb.simulator import simulator, coupling, integrators, monitors

    integ = integrators.HeunDeterministic(dt=0.05)
    mon = monitors.Raw(period=1.0)

    sim = simulator.Simulator(
        connectivity=conn,
        model=model,
        coupling=coupling.Linear(a=np.array([preset["coupling_a"]])),
        integrator=integ,
        monitors=[mon],
    )
    sim.configure()

    logger.info("Running baseline simulation (3000ms)...")
    start = time.time()

    results = []
    for (t, data), in sim(simulation_length=3000.0):
        if data is not None:
            results.append(data.copy())

    elapsed = time.time() - start
    logger.info("Baseline completed in {:.1f}s", elapsed)

    if not results:
        raise RuntimeError("Baseline simulation produced no output")

    output = np.concatenate(results, axis=0)

    # Extract signal -- handle 3D or 4D output
    if output.ndim == 4:
        signal = output[:, 0, :, 0]
    elif output.ndim == 3:
        signal = output[:, :, 0]
    else:
        signal = output

    # Replace NaN/Inf
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute spectral metrics
    fs = 1000.0
    freqs = np.fft.fftfreq(signal.shape[0], d=1.0 / fs)
    power = np.abs(np.fft.fft(signal, axis=0)) ** 2

    theta_power = float(np.mean(power[(freqs >= 4) & (freqs <= 8), :]))
    gamma_power = float(np.mean(power[(freqs >= 30) & (freqs <= 80), :]))

    # Functional connectivity (correlation between regions)
    if signal.shape[1] > 1:
        fc_matrix = np.corrcoef(signal.T)
        fc_matrix = np.nan_to_num(fc_matrix)
        # Synchrony index: mean off-diagonal correlation
        mask = ~np.eye(fc_matrix.shape[0], dtype=bool)
        synchrony = float(np.mean(np.abs(fc_matrix[mask])))
    else:
        synchrony = 0.0

    # Hippocampal activity (mean power in hippocampal regions)
    hippo_indices = [i for i in range(34, 38) if i < signal.shape[1]]
    if hippo_indices:
        hippo_activity = float(np.mean(np.var(signal[:, hippo_indices], axis=0)))
    else:
        hippo_activity = float(np.mean(np.var(signal, axis=0)))

    metrics = {
        "theta_power": theta_power,
        "gamma_power": gamma_power,
        "synchrony": synchrony,
        "hippo_activity": hippo_activity,
        "simulation_time_seconds": elapsed,
        "n_time_steps": output.shape[0],
    }

    logger.info(
        "Baseline metrics: theta={:.2e}, gamma={:.2e}, sync={:.3f}",
        theta_power, gamma_power, synchrony,
    )

    return metrics


def build_patient_twin(params: PatientParams) -> PatientTwin:
    """Build a complete patient brain digital twin from MRI-extracted parameters.

    This is the main entry point for T-05. It chains together:
        PatientParams -> Connectivity + JansenRit -> Baseline Simulation -> PatientTwin

    The returned PatientTwin contains everything needed to run drug simulations:
    the connectivity matrix, model configuration, and baseline metrics.
    """
    logger.info(
        "Building twin: disease={}, age={}, hippo_vol={:.2f}",
        params.disease_state, params.patient_age, params.hippocampal_volume_normalized,
    )

    # Get disease preset
    preset = DISEASE_PRESETS.get(params.disease_state, DISEASE_PRESETS["MCI"])

    # Build connectivity
    conn = _load_and_scale_connectivity(params)

    # Build model
    model = _build_model(params, preset)

    # Compute connectivity checksum for reproducibility tracking
    weights_bytes = conn.weights.tobytes()
    checksum = hashlib.sha256(weights_bytes).hexdigest()[:16]

    # Run baseline simulation
    baseline_metrics = _run_baseline_simulation(conn, model, preset)

    twin = PatientTwin(
        patient_params=params,
        connectivity_checksum=checksum,
        disease_preset=preset["preset_name"],
        baseline_metrics=baseline_metrics,
    )

    logger.info("Twin built: id={}, preset={}, checksum={}", twin.twin_id, twin.disease_preset, checksum)
    return twin


def run_twin_builder_test():
    """Test twin building with synthetic patient params. Requires TVB installed."""
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]SOMA Brain -- Twin Builder Test (T-05)[/bold cyan]\n")

    # Moderate AD patient
    params = PatientParams(
        hippocampal_volume_normalized=0.60,
        entorhinal_cortex_volume_normalized=0.55,
        prefrontal_volume_normalized=0.85,
        whole_brain_volume_normalized=0.78,
        global_fa_score=0.37,
        hippocampal_cingulum_fa=0.31,
        connectivity_scale_factor=0.70,
        hippocampal_pfc_connection_strength=0.55,
        disease_state="moderate_AD",
        estimated_amyloid_burden=0.65,
        patient_age=74,
        patient_sex="F",
        extraction_confidence=0.88,
        missing_fields=[],
    )

    console.print("  Building twin for moderate AD patient (74F)...")
    twin = build_patient_twin(params)

    console.print(f"  Twin ID: {twin.twin_id}")
    console.print(f"  Preset: {twin.disease_preset}")
    console.print(f"  Checksum: {twin.connectivity_checksum}")
    console.print(f"  Baseline metrics:")
    for key, val in twin.baseline_metrics.items():
        if isinstance(val, float):
            console.print(f"    {key}: {val:.4e}")
        else:
            console.print(f"    {key}: {val}")

    # Basic validation
    assert twin.baseline_metrics["theta_power"] > 0, "Theta power should be positive"
    assert twin.baseline_metrics["n_time_steps"] > 100, "Should have many time steps"
    assert twin.disease_preset == "moderate_AD_v1"

    console.print("\n[bold green]Twin builder test PASSED[/bold green]")
    return True


if __name__ == "__main__":
    run_twin_builder_test()
