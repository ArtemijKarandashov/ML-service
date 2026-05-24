from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select,delete, desc
from datetime import datetime
from fastapi import HTTPException, status

from .db import PklFileEntry


class PklFileService:

    async def get_pkl_file_entry(
            self, 
            session: AsyncSession, 
            uid: str
        ):
        stmt = select(PklFileEntry).where(PklFileEntry.uid == uid)
        result = await session.exec(stmt)
        user_data = result.first()

        if user_data is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PklFileEntry not found")

        return user_data


    async def pkl_file_entry_exists(
            self, 
            session: AsyncSession, 
            uid: str
        ):
        stmt = select(PklFileEntry).where(PklFileEntry.uid == uid)
        result = await session.exec(stmt)
        user_data = result.first()

        return not user_data is None


    async def create_pkl_file_entry(
            self, 
            session: AsyncSession, 
            uid: str,
            file_path: str,
            name: str
        ):

        new_file = PklFileEntry(
            uid = uid,
            filepath=file_path,
            name=name
        )
    
        session.add(new_file)
        await session.commit()
        return new_file
    

    async def delete_pkl_file_entry(
            self, 
            session: AsyncSession, 
            uid: str
        ):

        if not await self.pkl_file_entry_exists(session, uid):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PklFile for given uid not found.") 

        stmt = delete(PklFileEntry).where(PklFileEntry.uid == uid)
        result = await session.exec(stmt)
        await session.commit()

        return {"message": f"Successfully deleted PklFileEntry <{uid}>."}