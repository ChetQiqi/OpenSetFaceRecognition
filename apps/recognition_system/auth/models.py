import enum
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from .db import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    developer = "developer"
    viewer = "viewer"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default=UserRole.viewer.value)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
