"""
SOMA Brain — FastAPI Dependency Injection

Provides database sessions, auth verification, and audit logging
as injectable dependencies for route handlers.
"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from soma.config import get_settings

security = HTTPBearer()


def create_access_token(user_id: str) -> str:
    """Create a JWT access token for a user."""
    settings = get_settings()
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiry_hours)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """Validate JWT and return user_id. Raises 401 if invalid."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
