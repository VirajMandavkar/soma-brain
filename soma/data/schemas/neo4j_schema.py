"""
SOMA Brain — Neo4j Knowledge Graph Schema

Defines all node labels, relationship types, and indexes.
Run via scripts/setup_databases.py to initialize a fresh Neo4j instance.

Why Neo4j: drug-mechanism-finding relationships are inherently a graph problem.
Querying "which drugs target mechanisms that contradict recent findings"
is a 2-hop traversal in Neo4j but a 4-table JOIN nightmare in SQL.
"""

# Cypher statements to create constraints and indexes.
# Constraints enforce uniqueness (e.g., one node per SMILES) and
# automatically create indexes for fast lookups.

CONSTRAINTS = [
    "CREATE CONSTRAINT compound_smiles IF NOT EXISTS FOR (c:Compound) REQUIRE c.smiles IS UNIQUE",
    "CREATE CONSTRAINT paper_doi IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
    "CREATE CONSTRAINT mechanism_name IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT brain_region_name IF NOT EXISTS FOR (b:BrainRegion) REQUIRE b.name IS UNIQUE",
    "CREATE CONSTRAINT patient_twin_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.twin_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX compound_status IF NOT EXISTS FOR (c:Compound) ON (c.status)",
    "CREATE INDEX finding_confidence IF NOT EXISTS FOR (f:Finding) ON (f.confidence)",
    "CREATE INDEX paper_date IF NOT EXISTS FOR (p:Paper) ON (p.date)",
    "CREATE INDEX paper_relevance IF NOT EXISTS FOR (p:Paper) ON (p.relevance_score)",
]

# Example Cypher for seeding the 80 TVB brain regions.
# Allen Brain Atlas region names mapped to TVB indices.
SEED_BRAIN_REGIONS = """
UNWIND $regions AS r
MERGE (b:BrainRegion {name: r.name})
SET b.atlas_id = r.atlas_id,
    b.hemisphere = r.hemisphere
"""
