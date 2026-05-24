from fastapi import UploadFile
from urllib.parse import urljoin

import requests
import asyncio

from .config import Config


class StorageServiceAPI():

    def __init__(self, service_host: str):
        self._api_prefix = "/api/v1/storage"
        self._service_host: str = service_host


    async def save_pkl_file(self, file: UploadFile):
        url = urljoin(self._service_host, f"{self._api_prefix}/store")
        files = {"pkl_file": (file.filename, file.file, file.content_type)}
        response = await asyncio.to_thread(requests.post, url, files=files)
        return response

    
    async def get_pkl_file(self, uid: str):
        url = urljoin(self._service_host, f"{self._api_prefix}/get/{uid}")
        response = await asyncio.to_thread(requests.get, url)
        return response
    
    
    async def delete_pkl_file(self, uid: str):
        url = urljoin(self._service_host, f"{self._api_prefix}/delete/{uid}")
        response = await asyncio.to_thread(requests.delete, url)
        return response
    
storage_service = StorageServiceAPI(service_host=Config.STORAGE_SERVICE_API)