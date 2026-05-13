"""FastAPI entrypoint for the EEIE service."""

from __future__ import annotations

from fastapi import FastAPI

from eeie import __version__
from eeie.api.routers import battery, behavior, cost, evaluation, fleet, optimization, vehicles
from eeie.api.routers import range as range_router
from eeie.api.schemas import HealthResponse
from eeie.features import FEATURE_VERSION

app = FastAPI(
    title="EV Energy Intelligence Engine API",
    version=__version__,
    description=(
        "Predict, optimize, and explain EV energy behaviour. Every prediction "
        "returns a structured Insight (top features, financial/battery impact, "
        "confidence) alongside the numeric result."
    ),
)


app.include_router(range_router.router)
app.include_router(optimization.router)
app.include_router(battery.router)
app.include_router(cost.router)
app.include_router(behavior.router)
app.include_router(vehicles.router)
app.include_router(fleet.router)
app.include_router(evaluation.router)


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        feature_version=FEATURE_VERSION,
    )


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": "EEIE",
        "version": __version__,
        "docs": "/docs",
        "health": "/healthz",
    }
