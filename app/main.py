"""FastAPI entrypoint for the pattern scanner service."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .data_sources import build_coingecko_auth
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
    version="1.1.0",
    description=(
        "Scans crypto assets for the crown-shelf-right-spike base structure. "
        "Supports both symbol-based resolution and direct CoinGecko ids."
    ),
    servers=[{"url": BASE_URL}],
)


@app.on_event("startup")
async def validate_integrations() -> None:
    build_coingecko_auth()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="crypto-pattern-scanner",
        version="1.1.0",
    )


@app.post(
    "/scan",
    response_model=ScanResponse | CompactScanResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Scan assets by symbols and/or coingecko_ids",
)
async def scan(req: ScanRequest):
    if req.top_k > req.max_coins_to_evaluate:
        return JSONResponse(
            status_code=400,
            content={"detail": "top_k cannot be greater than max_coins_to_evaluate"},
        )

    if not req.symbols and not req.coingecko_ids and req.max_coins_to_evaluate < 1:
        return JSONResponse(
            status_code=400,
            content={"detail": "Provide symbols, coingecko_ids, or a positive max_coins_to_evaluate"},
        )

    return await scan_pattern(req)


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        message="Crypto Pattern Scanner API",
        docs="/docs",
        health="/health",
        openapi="/openapi.json",
        default_min_age_days=settings.default_min_age_days,
        default_max_age_days=settings.default_max_age_days,
    )
