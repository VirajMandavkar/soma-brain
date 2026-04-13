"""SOMA Brain — Knowledge Graph API Routes"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class KnowledgeQuery(BaseModel):
    cypher_query: str | None = None
    natural_language: str | None = None


@router.post("/query")
async def query_knowledge_graph(request: KnowledgeQuery):
    """Query the Neo4j knowledge graph via Cypher or natural language."""
    # Will be implemented in T-17 (Neo4j client)
    return {"nodes": [], "relationships": [], "message": "Not yet implemented — see T-17"}
