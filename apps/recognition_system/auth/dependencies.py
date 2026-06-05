from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from .db import get_auth_db
from .models import User
from .repository import UserRepository
from .service import AuthService


def get_user_repository(db: Session = Depends(get_auth_db)) -> UserRepository:
    return UserRepository(db)


def get_auth_service(repo: UserRepository = Depends(get_user_repository)) -> AuthService:
    return AuthService(repo)


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少 Bearer token")
    return auth_service.get_current_user_from_token(token)


def get_optional_current_user(
    authorization: Optional[str] = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[User]:
    token = _extract_bearer_token(authorization)
    if not token:
        return None
    return auth_service.get_current_user_from_token(token)


def require_roles(*allowed_roles: str) -> Callable:
    def checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足，需要角色: {', '.join(allowed_roles)}",
            )
        return current_user

    return checker
