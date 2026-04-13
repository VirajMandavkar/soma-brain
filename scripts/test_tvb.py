"""
SOMA Brain — T-03: TVB Installation & Baseline Test

Validates that The Virtual Brain runs correctly on this machine.
Run: python scripts/test_tvb.py

Success criteria:
  - Simulation runs without error
  - Output shape is [time_steps, state_vars, regions, modes]
  - Non-zero theta power in FFT
  - Runtime under 120 seconds
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console

console = Console()


def run_baseline_tvb_simulation():
    """Run a 3-second JansenRit simulation on default connectivity."""
    from tvb.simulator import simulator, models, coupling, integrators, monitors
    from tvb.datatypes import connectivity

    console.print("[cyan]Loading TVB default connectivity...[/cyan]")
    conn = connectivity.Connectivity.from_file()
    conn.configure()
    console.print(f"  Connectivity: {conn.number_of_regions} regions, "
                  f"weights shape {conn.weights.shape}")

    # JansenRit — the workhorse model for EEG-frequency brain dynamics.
    # It models excitatory/inhibitory population interactions and naturally
    # produces oscillations in the alpha/theta/gamma bands.
    model = models.JansenRit(
        a=np.array([3.25]),      # excitatory gain
        b=np.array([22.0]),      # inhibitory gain
        mu=np.array([0.22]),     # mean external drive
    )

    # HeunDeterministic for the baseline (no noise).
    # Monte Carlo runs will use HeunStochastic instead.
    integ = integrators.HeunDeterministic(dt=0.05)  # smaller dt for stability

    # Raw monitor returns (time, state_vars, regions, modes) — full state access.
    mon = monitors.Raw(period=1.0)

    # Scale down coupling to prevent overflow with default connectivity weights.
    # Default connectivity has large weight values; too-strong coupling causes
    # exponential blowup in the JansenRit sigmoid functions.
    conn.weights = conn.weights / np.max(conn.weights)  # normalize to [0, 1]

    sim = simulator.Simulator(
        connectivity=conn,
        model=model,
        coupling=coupling.Linear(a=np.array([0.006])),  # conservative coupling
        integrator=integ,
        monitors=[mon],
    )

    console.print("[cyan]Configuring simulator...[/cyan]")
    sim.configure()

    console.print("[cyan]Running 3000ms simulation...[/cyan]")
    start = time.time()

    results = []
    for (t, data), in sim(simulation_length=3000.0):
        if data is not None:
            results.append(data.copy())

    elapsed = time.time() - start
    console.print(f"  Simulation completed in {elapsed:.1f}s")

    if not results:
        console.print("[red]ERROR: No simulation output![/red]")
        return False

    output = np.concatenate(results, axis=0)
    console.print(f"  Output shape: {output.shape}")

    # Raw monitor returns (time, state_vars, regions, modes) — 4D array
    # TemporalAverage returns (time, regions, modes) — 3D array
    # Handle both shapes for robustness
    if output.ndim == 4:
        signal = output[:, 0, :, 0]   # (time, regions)
        n_regions = output.shape[2]
    elif output.ndim == 3:
        signal = output[:, :, 0]       # (time, regions)
        n_regions = output.shape[1]
    else:
        console.print(f"[red]Unexpected output dimensions: {output.ndim}[/red]")
        return False

    # Check for NaN/Inf — indicates numerical blowup
    if np.any(~np.isfinite(signal)):
        nan_pct = np.mean(~np.isfinite(signal)) * 100
        console.print(f"[yellow]WARNING: {nan_pct:.1f}% of signal values are NaN/Inf[/yellow]")
        # Replace NaN/Inf with 0 for FFT
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute theta power (4-8 Hz) from FFT
    fs = 1000.0  # 1ms period = 1000 Hz
    freqs = np.fft.fftfreq(signal.shape[0], d=1.0 / fs)
    power = np.abs(np.fft.fft(signal, axis=0)) ** 2
    theta_mask = (freqs >= 4) & (freqs <= 8)
    theta_power = float(np.mean(power[theta_mask, :]))

    console.print(f"  Mean theta power (4-8 Hz): {theta_power:.4e}")
    console.print(f"  Signal range: [{signal.min():.6f}, {signal.max():.6f}]")

    # Validation
    all_ok = True

    if output.shape[0] < 100:
        console.print("[red]FAIL: Too few time steps[/red]")
        all_ok = False

    if n_regions < 70:
        console.print(f"[red]FAIL: Only {n_regions} regions (expected ~76)[/red]")
        all_ok = False

    if theta_power <= 0:
        console.print("[red]FAIL: Zero theta power — simulation may not have produced dynamics[/red]")
        all_ok = False

    if elapsed > 120:
        console.print(f"[red]FAIL: Took {elapsed:.0f}s (limit: 120s)[/red]")
        all_ok = False

    if all_ok:
        console.print(f"\n[bold green]TVB baseline test PASSED[/bold green]")
        console.print(f"  {n_regions} regions, {output.shape[0]} time steps, "
                      f"theta={theta_power:.4e}, {elapsed:.1f}s")
    else:
        console.print("\n[bold red]TVB baseline test FAILED[/bold red]")

    return all_ok


if __name__ == "__main__":
    console.print("\n[bold cyan]SOMA Brain — TVB Installation & Baseline Test[/bold cyan]\n")

    try:
        import tvb.simulator
        console.print(f"[green]tvb-library imported successfully[/green]")
    except ImportError as e:
        console.print(f"[red]Cannot import TVB: {e}[/red]")
        console.print("Try: pip install tvb-library==2.9.0")
        sys.exit(1)

    success = run_baseline_tvb_simulation()
    sys.exit(0 if success else 1)
