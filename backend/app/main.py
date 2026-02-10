"""
Eye Hear U - Backend API
FastAPI application for ASL sign recognition inference.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, predict

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API for ASL-to-English sign translation. Accepts images and returns predicted sign labels.",
    version="0.1.0",
)

# CORS middleware so the mobile app can reach us
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])


@app.on_event("startup")
async def startup_event():
    """Load ML model into memory on startup."""
    # TODO: Load the trained model here so it's ready for inference
    # from app.services.model_service import load_model
    # app.state.model = load_model(settings.model_path, settings.model_device)
    print(f"[startup] {settings.app_name} is ready.")
