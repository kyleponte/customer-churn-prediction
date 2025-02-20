import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from src.api.endpoints import router as api_router

# Re-export the FastAPI app
__all__ = ['app']

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Include the API router
app.include_router(api_router, prefix="/api")

# Templates
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 