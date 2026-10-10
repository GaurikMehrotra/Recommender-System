"""FastAPI application factory; explicit artifact location through RECOMMENDER_ARTIFACTS."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommend import Recommender


class Request(BaseModel):
    """Internal zero-based IDs; send either user or liked IDs."""

    user: int | None = None
    liked: list[int] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=100)


def create_app(artifacts: str | None = None) -> FastAPI:
    """Load immutable retrieval artifacts during service lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.recommender = Recommender(
            artifacts or os.environ.get("RECOMMENDER_ARTIFACTS", "artifacts/demo")
        )
        yield

    app = FastAPI(title="MovieLens Recommender", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "catalog_size": len(app.state.recommender.titles)}

    @app.post("/recommend")
    def recommend(request: Request) -> list[dict]:
        try:
            return app.state.recommender.recommend(request.user, request.liked, request.k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
