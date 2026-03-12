from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import settings
from .models import ErrorResponse, ScanRequest, ScanResponse
from .services import scan_pattern

app = FastAPI(
    title="Crypto Pattern Scanner API",
    version="1.0.0",
    description="Scans recently listed crypto assets and ranks them by similarity to the crown-shelf-right-spike daily-chart silhouette.",
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "crypto-pattern-scanner",
        "version": "1.0.0",
    }


@app.post(
    "/scan",
    response_model=ScanResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def scan(req: ScanRequest):
    if req.top_k > req.max_coins_to_evaluate:
        return JSONResponse(
            status_code=400,
            content={"detail": "top_k cannot be greater than max_coins_to_evaluate"},
        )

    return await scan_pattern(req)


@app.get("/")
async def root() -> dict:
    return {
        "message": "Crypto Pattern Scanner API",
        "docs": "/docs",
        "health": "/health",
        "default_min_age_days": settings.default_min_age_days,
        "default_max_age_days": settings.default_max_age_days,
    }
