from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import text

from src.config import Config


engine = create_async_engine(
        url=Config.SQLITE_URL,
        echo=True,
    )

async def init_db():
    async with engine.begin() as conn:
        from .models import PklFileEntry

        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncSession: # type: ignore
    Session = sessionmaker(
        bind=engine, 
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with Session() as session:
        yield session


async def check_database_health() -> bool:
    """
    Проверяет работоспособность БД, выполняя запрос SELECT 1.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False