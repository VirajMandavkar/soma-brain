"""
SOMA Brain -- T-09: Drug-to-TVB Perturbation Translation

Translates drug binding data (mechanism type + binding affinity + receptor
expression) into TVB JansenRit parameter deltas.

This is the biophysical bridge between molecular docking and brain network
simulation. When a drug binds BACE1 with -8.5 kcal/mol in the hippocampus,
what does that DO to neural dynamics? This module answers that question.

Each mechanism type maps to specific JansenRit parameters:
  - a: excitatory gain (PSP amplitude, excitatory synapse strength)
  - b: inhibitory gain (inhibitory PSP amplitude)
  - mu: external input drive (thalamocortical drive)
  - coupling_a: long-range coupling strength
  - noise_nsig: neural noise amplitude

The mapping is based on published computational neuropharmacology literature:
  - Moran et al. (2011) NeuroImage — DCM pharmacology
  - Bojak & Liley (2005) — mean-field anesthesia modeling
  - Breakspear et al. (2006) — neural mass pharmacology
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class ParameterDelta:
    """Change to TVB JansenRit parameters from a single drug mechanism."""
    mechanism: str
    target: str
    binding_kcal: float

    # Multiplicative factors (1.0 = no change)
    a_factor: float = 1.0       # excitatory gain
    b_factor: float = 1.0       # inhibitory gain
    mu_factor: float = 1.0      # external drive
    coupling_factor: float = 1.0  # long-range coupling

    # Regional specificity: region_index -> factor_override
    # If empty, the effect is global. If populated, only those regions are affected.
    region_factors: dict[int, float] = field(default_factory=dict)

    reasoning: str = ""


def _binding_strength(binding_kcal: float) -> float:
    """Convert binding energy (kcal/mol) to a 0-1 effect strength.

    More negative = stronger binding = larger effect.
    Typical range: -5 (weak) to -12 (very strong).
    We use a sigmoid-like mapping that saturates at extreme values —
    because binding affinity has diminishing returns on pharmacological effect.
    """
    # Shift so -7.0 maps to ~0.5 (moderate binding)
    x = -(binding_kcal + 7.0) / 2.0
    strength = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(strength, 0.05, 0.95))


def _scale_by_expression(base_effect: float, expression_level: float) -> float:
    """Scale drug effect by receptor expression level in a brain region.

    If a receptor isn't expressed in a region, the drug can't act there.
    expression_level: 0 (not expressed) to 1 (maximum expression).
    """
    return base_effect * expression_level


# ============================================================
# Mechanism-specific translation functions
# ============================================================

def _bace1_inhibition(binding_kcal: float, region_expression: dict[int, float]) -> ParameterDelta:
    """BACE1 inhibition -> reduces amyloid-driven excitatory imbalance.

    BACE1 cleaves APP to produce amyloid-beta. Excess amyloid-beta causes
    excitatory/inhibitory imbalance (hyperexcitability). Inhibiting BACE1
    reduces amyloid, which should normalize the E/I balance.

    Effect: reduce excitatory gain (a) toward healthy levels.
    This is a SLOW mechanism — takes weeks to months in reality. In simulation,
    we model the equilibrium state (what happens after amyloid clears).
    """
    strength = _binding_strength(binding_kcal)

    # Reduce excitatory gain (bringing hyperexcitable AD brain toward normal)
    # Strong binding -> a_factor closer to 0.85 (15% reduction in excitatory gain)
    a_reduction = 0.15 * strength
    a_factor = 1.0 - a_reduction

    return ParameterDelta(
        mechanism="BACE1_inhibition",
        target="BACE1",
        binding_kcal=binding_kcal,
        a_factor=a_factor,
        region_factors={r: _scale_by_expression(a_reduction, e)
                       for r, e in region_expression.items()},
        reasoning=f"BACE1 inhibition (strength={strength:.2f}): reduces amyloid-driven "
                  f"excitatory imbalance by {a_reduction*100:.1f}% at equilibrium",
    )


def _gaba_potentiation(binding_kcal: float, region_expression: dict[int, float]) -> ParameterDelta:
    """GABA-A receptor potentiation -> increases inhibitory gain.

    Benzodiazepines, barbiturates, and some anesthetics work this way.
    In AD, GABAergic interneurons are lost, reducing inhibition.
    Potentiating remaining GABA receptors partially compensates.

    Effect: increase inhibitory gain (b).
    """
    strength = _binding_strength(binding_kcal)

    b_increase = 0.20 * strength  # up to 20% increase in inhibitory gain
    b_factor = 1.0 + b_increase

    return ParameterDelta(
        mechanism="GABA_potentiation",
        target="GABA_A",
        binding_kcal=binding_kcal,
        b_factor=b_factor,
        region_factors={r: _scale_by_expression(b_increase, e)
                       for r, e in region_expression.items()},
        reasoning=f"GABA potentiation (strength={strength:.2f}): increases inhibitory "
                  f"gain by {b_increase*100:.1f}%, compensating for AD interneuron loss",
    )


def _nmda_modulation(binding_kcal: float, region_expression: dict[int, float]) -> ParameterDelta:
    """NMDA receptor modulation -> alters coupling and excitatory gain.

    Memantine (an NMDA antagonist) is the prototype. It blocks excessive
    glutamatergic activation while preserving normal signaling (voltage-dependent
    block at the channel). In TVB terms: reduces coupling (long-range glutamatergic)
    and slightly reduces excitatory gain.

    Effect: reduce coupling_a and slightly reduce a.
    """
    strength = _binding_strength(binding_kcal)

    coupling_reduction = 0.12 * strength
    a_reduction = 0.08 * strength

    return ParameterDelta(
        mechanism="NMDA_modulation",
        target="NMDAR_GluN2B",
        binding_kcal=binding_kcal,
        a_factor=1.0 - a_reduction,
        coupling_factor=1.0 - coupling_reduction,
        region_factors={r: _scale_by_expression(coupling_reduction, e)
                       for r, e in region_expression.items()},
        reasoning=f"NMDA modulation (strength={strength:.2f}): reduces excitotoxic coupling "
                  f"by {coupling_reduction*100:.1f}% and excitatory gain by {a_reduction*100:.1f}%",
    )


def _ache_inhibition(binding_kcal: float, region_expression: dict[int, float]) -> ParameterDelta:
    """AChE inhibition -> increases cholinergic drive.

    Donepezil, rivastigmine, galantamine. Inhibiting acetylcholinesterase
    increases acetylcholine at synapses. In the brain, this increases
    thalamocortical drive (mu parameter) and modestly enhances coupling.

    Effect: increase mu (external drive) and slightly increase coupling.
    """
    strength = _binding_strength(binding_kcal)

    mu_increase = 0.15 * strength
    coupling_increase = 0.05 * strength

    return ParameterDelta(
        mechanism="AChE_inhibition",
        target="AChE",
        binding_kcal=binding_kcal,
        mu_factor=1.0 + mu_increase,
        coupling_factor=1.0 + coupling_increase,
        region_factors={r: _scale_by_expression(mu_increase, e)
                       for r, e in region_expression.items()},
        reasoning=f"AChE inhibition (strength={strength:.2f}): boosts cholinergic drive "
                  f"by {mu_increase*100:.1f}% and coupling by {coupling_increase*100:.1f}%",
    )


def _mglur5_antagonism(binding_kcal: float, region_expression: dict[int, float]) -> ParameterDelta:
    """mGluR5 antagonism -> reduces excitatory gain.

    mGluR5 is a metabotropic glutamate receptor. Antagonizing it reduces
    excitatory postsynaptic potentials. In AD, mGluR5 mediates synaptotoxic
    signaling from amyloid-beta oligomers, so blocking it is neuroprotective.

    Effect: reduce excitatory gain (a) without affecting inhibition.
    """
    strength = _binding_strength(binding_kcal)

    a_reduction = 0.10 * strength

    return ParameterDelta(
        mechanism="mGluR5_antagonism",
        target="mGluR5",
        binding_kcal=binding_kcal,
        a_factor=1.0 - a_reduction,
        reasoning=f"mGluR5 antagonism (strength={strength:.2f}): reduces excitatory "
                  f"gain by {a_reduction*100:.1f}%, neuroprotective against amyloid synaptotoxicity",
    )


# Registry of all supported mechanisms
MECHANISM_MAP = {
    "BACE1_inhibition": _bace1_inhibition,
    "GABA_potentiation": _gaba_potentiation,
    "NMDA_modulation": _nmda_modulation,
    "AChE_inhibition": _ache_inhibition,
    "mGluR5_antagonism": _mglur5_antagonism,
}


def translate_drug_to_tvb_delta(
    mechanism: str,
    binding_kcal: float,
    region_expression: dict[int, float] | None = None,
) -> ParameterDelta:
    """Main entry point: translate a drug mechanism into TVB parameter changes.

    Args:
        mechanism: One of the supported mechanism types (see MECHANISM_MAP).
        binding_kcal: Docking binding energy (more negative = stronger).
        region_expression: Map of region_index -> expression_level (0-1).
                          If None, assumes uniform expression across all regions.

    Returns:
        ParameterDelta with multiplicative factors for JansenRit parameters.
    """
    if mechanism not in MECHANISM_MAP:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. "
            f"Supported: {list(MECHANISM_MAP.keys())}"
        )

    if region_expression is None:
        # Default: uniform expression in all 76 regions
        region_expression = {i: 0.5 for i in range(76)}

    delta = MECHANISM_MAP[mechanism](binding_kcal, region_expression)
    logger.info("Perturbation: {} -> a={:.3f}, b={:.3f}, mu={:.3f}, coupling={:.3f}",
                mechanism, delta.a_factor, delta.b_factor, delta.mu_factor, delta.coupling_factor)
    return delta


def combine_deltas(deltas: list[ParameterDelta]) -> ParameterDelta:
    """Combine multiple drug mechanism effects into a single delta.

    When a drug has multiple mechanisms (e.g., hits both BACE1 and mGluR5),
    we multiply the factors together. This is a simplification — real
    pharmacology has nonlinear interactions — but it's the standard
    approach in computational pharmacology modeling.
    """
    combined = ParameterDelta(
        mechanism="+".join(d.mechanism for d in deltas),
        target="+".join(d.target for d in deltas),
        binding_kcal=min(d.binding_kcal for d in deltas),
    )

    for d in deltas:
        combined.a_factor *= d.a_factor
        combined.b_factor *= d.b_factor
        combined.mu_factor *= d.mu_factor
        combined.coupling_factor *= d.coupling_factor

    combined.reasoning = " | ".join(d.reasoning for d in deltas)
    return combined


def run_perturbation_tests():
    """Validate all 5 mechanism types produce correct directional effects."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]SOMA Brain -- Perturbation Translation Test (T-09)[/bold cyan]\n")

    # Default region expression
    expr = {i: 0.6 for i in range(76)}

    test_cases = [
        # (mechanism, binding_kcal, expected_direction)
        ("BACE1_inhibition", -8.5, {"a": "decrease"}),
        ("GABA_potentiation", -7.2, {"b": "increase"}),
        ("NMDA_modulation", -6.8, {"a": "decrease", "coupling": "decrease"}),
        ("AChE_inhibition", -9.0, {"mu": "increase", "coupling": "increase"}),
        ("mGluR5_antagonism", -7.5, {"a": "decrease"}),
    ]

    table = Table(title="Perturbation Validation")
    table.add_column("Mechanism")
    table.add_column("Binding")
    table.add_column("a_factor")
    table.add_column("b_factor")
    table.add_column("mu_factor")
    table.add_column("coupling")
    table.add_column("Match")

    all_ok = True
    for mechanism, kcal, expected in test_cases:
        delta = translate_drug_to_tvb_delta(mechanism, kcal, expr)

        # Check directions
        ok = True
        for param, direction in expected.items():
            if param == "a" and direction == "decrease" and delta.a_factor >= 1.0:
                ok = False
            if param == "b" and direction == "increase" and delta.b_factor <= 1.0:
                ok = False
            if param == "mu" and direction == "increase" and delta.mu_factor <= 1.0:
                ok = False
            if param == "coupling" and direction == "decrease" and delta.coupling_factor >= 1.0:
                ok = False
            if param == "coupling" and direction == "increase" and delta.coupling_factor <= 1.0:
                ok = False

        all_ok = all_ok and ok
        table.add_row(
            mechanism, f"{kcal}", f"{delta.a_factor:.3f}", f"{delta.b_factor:.3f}",
            f"{delta.mu_factor:.3f}", f"{delta.coupling_factor:.3f}",
            "[green]OK[/green]" if ok else "[red]FAIL[/red]",
        )

    console.print(table)

    if all_ok:
        console.print("\n[bold green]All perturbation tests PASSED[/bold green]")
    else:
        console.print("\n[bold red]Some tests FAILED[/bold red]")

    return all_ok


if __name__ == "__main__":
    run_perturbation_tests()
