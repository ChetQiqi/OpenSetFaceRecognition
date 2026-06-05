from typing import Optional

from fastapi import HTTPException, status

from .models import User, UserRole
from .repository import UserRepository
from .security import create_access_token, decode_access_token, hash_password, verify_password


class AuthService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def register_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        email: Optional[str] = None,
        current_user: Optional[User] = None,
    ) -> User:
        user_count = self.repository.count_users()

        if user_count == 0:
            assigned_role = UserRole.admin.value
        elif current_user is not None and current_user.role == UserRole.admin.value:
            assigned_role = role.value
        elif current_user is None:
            assigned_role = UserRole.viewer.value
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="仅管理员可以创建新用户",
            )

        if self.repository.get_by_username(username) is not None:
            raise HTTPException(status_code=400, detail="用户名已存在")
        if email and self.repository.get_by_email(email) is not None:
            raise HTTPException(status_code=400, detail="邮箱已存在")

        return self.repository.create_user(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role=assigned_role,
        )

    def update_user_role(self, user_id: int, new_role: UserRole, current_user: User) -> User:
        if current_user.id == user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="不能修改自己的角色",
            )
        user = self.repository.update_user(user_id, role=new_role.value)
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        return user

    def login(self, username: str, password: str):
        user = self.repository.get_by_username(username)
        if user is None or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="用户已被禁用")

        token = create_access_token(subject=user.username, role=user.role)
        return {"access_token": token, "token_type": "bearer", "role": user.role}

    def get_current_user_from_token(self, token: str) -> User:
        payload = decode_access_token(token)
        username = str(payload.get("sub", ""))
        if not username:
            raise HTTPException(status_code=401, detail="token 缺少用户信息")
        user = self.repository.get_by_username(username)
        if user is None:
            raise HTTPException(status_code=401, detail="token 用户不存在")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="用户已被禁用")
        return user
