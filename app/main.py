"""FastAPI entrypoint for the pattern scanner service."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models import CompactScanResponse, ErrorResponse, ScanRequest, ScanResponse
from .services import scan_pattern

BASE_URL = "https://pattern-scanner-api.onrender.com"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class RootResponse(BaseModel):
    message: str
    docs: str
    health: str
    openapi: str
    default_min_age_days: int
    default_max_age_days: int


app = FastAPI(
    title="Crypto Pattern Scanner API",
    version="1.0.0",
    description=(
        "Scans crypto assets for the crown-shelf-right-spike base structure. "
        "Supports both symbol-based resolution and direct CoinGecko ids."
    ),
    servers=[{"url": BASE_URL}],
)


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        message="Crypto Pattern Scanner API is running.",
        docs="/docs",
        health="/health",
        openapi="/openapi.json",
        default_min_age_days=14,
        default_max_age_days=90,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="crypto-pattern-scanner",
        version="1.0.0",
    )


@app.post(
    "/scan",
    response_model=ScanResponse | CompactScanResponse,
    responses={400: {"model": ErrorResponse}},
)
async def scan(req: ScanRequest):
    try:
        return await scan_pattern(req)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=str(exc)).model_dump(),
        )
