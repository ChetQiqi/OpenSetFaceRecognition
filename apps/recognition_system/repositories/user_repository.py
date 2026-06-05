from sqlalchemy.orm import Session

from apps.recognition_system.auth.repository import UserRepository as AuthUserRepository


class UserRepository(AuthUserRepository):
    """Repository 层：用户与角色管理（SQLAlchemy）。"""

    def __init__(self, db: Session):
        super().__init__(db)
