from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .models import UserRole


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: Optional[str] = None
    role: UserRole = UserRole.viewer


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: UserRole


class CurrentUserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


class UserListItem(BaseModel):
    id: int
    username: str
    email: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime
