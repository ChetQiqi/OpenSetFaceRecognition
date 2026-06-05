from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from apps.recognition_system.config import get_config


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
config = get_config()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str, role: str, expires_minutes: Optional[int] = None) -> str:
    expire_delta = timedelta(
        minutes=expires_minutes if expires_minutes is not None else config.jwt_access_token_expire_minutes
    )
    expire = datetime.now(timezone.utc) + expire_delta
    payload: Dict[str, object] = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, object]:
    try:
        return jwt.decode(token, config.jwt_secret_key, algorithms=[config.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("无效或已过期的 token") from exc
