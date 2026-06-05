from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def count_users(self) -> int:
        return len(self.db.execute(select(User.id)).all())

    def get_by_username(self, username: str) -> Optional[User]:
        return self.db.execute(select(User).where(User.username == username)).scalar_one_or_none()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    def get_by_id(self, user_id: int) -> Optional[User]:
        return self.db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()

    def list_users(self) -> List[User]:
        return list(self.db.execute(select(User).order_by(User.created_at.desc())).scalars().all())

    def create_user(self, username: str, email: Optional[str], password_hash: str, role: str) -> User:
        user = User(username=username, email=email, password_hash=password_hash, role=role)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete_user(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if user is None:
            return False
        self.db.delete(user)
        self.db.commit()
        return True

    def update_user(self, user_id: int, **fields) -> Optional[User]:
        user = self.get_by_id(user_id)
        if user is None:
            return None
        for key, value in fields.items():
            if hasattr(user, key):
                setattr(user, key, value)
        self.db.commit()
        self.db.refresh(user)
        return user
