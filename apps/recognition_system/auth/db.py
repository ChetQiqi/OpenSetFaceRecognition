from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from apps.recognition_system.config import get_config


config = get_config()

engine = create_engine(
    f"sqlite:///{config.auth_db_path}",
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_auth_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_auth_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
