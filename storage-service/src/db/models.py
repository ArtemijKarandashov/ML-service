from typing import List, Optional
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Column, Integer, DateTime, func

from datetime import datetime, date


class PklFileEntry(SQLModel, table=True):
        __tablename__ = "pkl_file_entry"

        uid: str = Field(primary_key=True, unique=True)
        filepath: str = Field(max_length=100)
        name: str = Field(max_length=50)

        created_at: datetime = Field(sa_column=Column(DateTime, server_default=func.now()))
        updated_at: datetime = Field(sa_column=Column(DateTime, onupdate=func.now()))