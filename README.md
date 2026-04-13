# SOMA Brain

**Synthetic Organismic Modelling and Analysis — Brain**

Enterprise AI platform for CNS drug discovery. Reads patient MRI reports, builds brain digital twins using The Virtual Brain, runs 20,000 Monte Carlo drug simulations, and predicts which candidates will work — for this specific patient.

Built for the **Gemma 4 Good Hackathon** (Google DeepMind, May 2026).

## The Problem

CNS drug development fails 99%+ of the time. $6.1B wasted annually on trials that fail because nobody predicted BBB penetration, receptor binding, or brain network dynamics beforehand.

## What SOMA Brain Does

1. **MRI Report → Patient Parameters** (Gemma 4 multimodal extraction)
2. **Patient Parameters → Brain Digital Twin** (The Virtual Brain, 80-region network)
3. **Drug Candidate → 20,000 Monte Carlo Simulations** (BBB + docking + TVB + PK variability)
4. **Probability Distributions** with 95% confidence intervals (not single-point guesses)
5. **Nightly Literature Ingestion** (PubMed + bioRxiv + Semantic Scholar → Neo4j knowledge graph)
6. **Regulatory-Grade Reports** (IND application format with full reasoning chains)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/soma-brain.git
cd soma-brain
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Start infrastructure
cd docker && docker compose up -d && cd ..

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Validate everything works
python scripts/validate_apis.py

# Run the API server
uvicorn soma.api.main:app --reload
```

## Architecture

7-layer system: External Data → Databases → Literature Engine → Simulation Engine → AI Reasoning → API → Frontend

See `docs/architecture.md` for full details.

## License

Apache 2.0
