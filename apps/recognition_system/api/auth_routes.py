from typing import List, Optional

from fastapi import APIRouter, Depends

from apps.recognition_system.auth.dependencies import (
    get_auth_service,
    get_current_user,
    get_optional_current_user,
    get_user_repository,
    require_roles,
)
from apps.recognition_system.auth.models import User, UserRole
from apps.recognition_system.auth.repository import UserRepository
from apps.recognition_system.auth.schemas import (
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateUserRoleRequest,
    UserListItem,
)
from apps.recognition_system.auth.service import AuthService


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, auth_service: AuthService = Depends(get_auth_service)):
    return auth_service.login(payload.username, payload.password)


@router.post("/register", response_model=CurrentUserResponse)
def register(
    payload: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    user = auth_service.register_user(
        username=payload.username,
        password=payload.password,
        role=payload.role,
        email=payload.email,
        current_user=current_user,
    )
    return CurrentUserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/me", response_model=CurrentUserResponse)
def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=UserRole(current_user.role),
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


@router.get("/users", response_model=List[UserListItem])
def list_users(
    _: User = Depends(require_roles(UserRole.admin.value)),
    repo: UserRepository = Depends(get_user_repository),
):
    users = repo.list_users()
    return [
        UserListItem(
            id=user.id,
            username=user.username,
            email=user.email,
            role=UserRole(user.role),
            is_active=user.is_active,
            created_at=user.created_at,
        )
        for user in users
    ]


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    current_user: User = Depends(require_roles(UserRole.admin.value)),
    repo: UserRepository = Depends(get_user_repository),
):
    if current_user.id == user_id:
        return {"deleted": False, "reason": "不能删除当前登录用户"}
    deleted = repo.delete_user(user_id)
    return {"deleted": deleted, "user_id": user_id}


@router.put("/users/{user_id}/role", response_model=UserListItem)
def update_user_role(
    user_id: int,
    payload: UpdateUserRoleRequest,
    current_user: User = Depends(require_roles(UserRole.admin.value)),
    auth_service: AuthService = Depends(get_auth_service),
):
    user = auth_service.update_user_role(user_id, payload.role, current_user)
    return UserListItem(
        id=user.id,
        username=user.username,
        email=user.email,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
    )
