"""
SOMA Brain — T-02: API Validation Script

Tests every external API and local service that SOMA Brain depends on.
Run: python scripts/validate_apis.py

Each check is independent — if one fails, the others still run.
Fix all failures before proceeding to T-03.
"""

import sys
import json
from pathlib import Path

# Add project root to path so we can import soma.config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from rich.console import Console
from rich.table import Table

from soma.config import get_settings

console = Console()
settings = get_settings()

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    """Run a check function, catch all exceptions, record result."""
    try:
        msg = fn()
        results.append((name, True, msg))
    except Exception as e:
        results.append((name, False, str(e)[:120]))


# ── External APIs ──────────────────────────────────────────

def check_chembl():
    r = requests.get(
        "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL941.json",
        timeout=15,
    )
    r.raise_for_status()
    name = r.json().get("pref_name", "unknown")
    return f"ChEMBL OK — fetched {name}"


def check_pdb():
    r = requests.get(
        "https://data.rcsb.org/rest/v1/core/entry/2OHU",
        timeout=15,
    )
    r.raise_for_status()
    title = r.json().get("struct", {}).get("title", "unknown")[:60]
    return f"PDB OK — {title}"


def check_allen_brain():
    r = requests.get(
        "https://api.brain-map.org/api/v2/data/query.json?criteria=model::Gene,rma::criteria,[acronym$eq'BACE1']",
        timeout=15,
    )
    r.raise_for_status()
    n = r.json().get("num_rows", 0)
    return f"Allen Brain Atlas OK — {n} gene(s) found for BACE1"


def check_kegg():
    r = requests.get(
        "https://rest.kegg.jp/get/hsa:23621",
        timeout=15,
    )
    r.raise_for_status()
    return f"KEGG OK — {len(r.text)} chars returned for BACE1 (hsa:23621)"


def check_pubmed():
    api_key_param = f"&api_key={settings.ncbi_api_key}" if settings.ncbi_api_key else ""
    r = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term=alzheimer+drug+discovery&retmax=3&retmode=json{api_key_param}",
        timeout=15,
    )
    r.raise_for_status()
    count = r.json().get("esearchresult", {}).get("count", "0")
    return f"PubMed OK — {count} results for 'alzheimer drug discovery'"


def check_biorxiv():
    r = requests.get(
        "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-02/0/json",
        timeout=15,
    )
    r.raise_for_status()
    n = len(r.json().get("collection", []))
    return f"bioRxiv OK — {n} preprints in sample date range"


def check_semantic_scholar():
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search?query=alzheimer+BACE1&limit=3",
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    n = r.json().get("total", 0)
    return f"Semantic Scholar OK — {n} total results for 'alzheimer BACE1'"


# ── Docker Services ────────────────────────────────────────

def check_neo4j():
    r = requests.get("http://localhost:7474", timeout=5)
    return f"Neo4j OK — HTTP {r.status_code}"


def check_weaviate():
    r = requests.get("http://localhost:8080/v1/.well-known/ready", timeout=5)
    r.raise_for_status()
    return "Weaviate OK — ready"


def check_postgres():
    # Use psycopg2 or just try a TCP connect
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(("localhost", 5432))
    s.close()
    return "PostgreSQL OK — port 5432 accepting connections"


def check_redis():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(("localhost", 6379))
    s.close()
    return "Redis OK — port 6379 accepting connections"


# ── Python Libraries ───────────────────────────────────────

def check_rdkit():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert mol is not None
    return f"RDKit OK — parsed aspirin ({mol.GetNumAtoms()} atoms)"


def check_tvb():
    from tvb.simulator import models
    jr = models.JansenRit()
    return f"TVB OK — JansenRit model loaded ({len(jr.state_variables)} state vars)"


# ── Run all checks ─────────────────────────────────────────

if __name__ == "__main__":
    console.print("\n[bold cyan]SOMA Brain — API & Service Validation[/bold cyan]\n")

    checks = [
        ("ChEMBL API", check_chembl),
        ("PDB (RCSB)", check_pdb),
        ("Allen Brain Atlas", check_allen_brain),
        ("KEGG Pathways", check_kegg),
        ("PubMed E-utilities", check_pubmed),
        ("bioRxiv API", check_biorxiv),
        ("Semantic Scholar", check_semantic_scholar),
        ("Neo4j (Docker)", check_neo4j),
        ("Weaviate (Docker)", check_weaviate),
        ("PostgreSQL (Docker)", check_postgres),
        ("Redis (Docker)", check_redis),
        ("RDKit", check_rdkit),
        ("TVB (tvb-library)", check_tvb),
    ]

    for name, fn in checks:
        console.print(f"  Checking {name}...", end=" ")
        check(name, fn)
        _, ok, msg = results[-1]
        if ok:
            console.print(f"[green]\u2713[/green] {msg}")
        else:
            console.print(f"[red]\u2717[/red] {msg}")

    # Summary table
    console.print()
    table = Table(title="Validation Summary")
    table.add_column("Service", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    passed = 0
    for name, ok, msg in results:
        status = "[green]\u2713 PASS[/green]" if ok else "[red]\u2717 FAIL[/red]"
        table.add_row(name, status, msg[:80])
        if ok:
            passed += 1

    console.print(table)
    console.print(f"\n[bold]{passed}/{len(results)} checks passed.[/bold]")

    if passed < len(results):
        console.print("[yellow]Fix failing checks before proceeding to T-03.[/yellow]")
        sys.exit(1)
    else:
        console.print("[green]All checks passed! Ready for T-03.[/green]")
        sys.exit(0)
