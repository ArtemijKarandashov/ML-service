from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

initial_router = APIRouter()
templates = Jinja2Templates(directory="templates")

@initial_router.get("/", response_class=HTMLResponse)
async def root(
    request: Request
):
    return templates.TemplateResponse(request=request, name="index.html")