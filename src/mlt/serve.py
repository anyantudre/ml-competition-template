"""A small HTTP API around a finished run (needs the `serve` extra).

    mlt serve outputs/my_run --port 8000
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"records": [{"feature_a": 1.0, "feature_b": "x"}]}'

The models load once at start-up and every fold model votes, as in the submission.
(No `from __future__ import annotations` in this file: FastAPI must see the request model type.)
"""

import time

import pandas as pd

from .predict import Predictor
from .utils import read_json, require


def create_app(run_dir: str):
    require("fastapi", "serve")
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    predictor = Predictor(run_dir)
    info = read_json(predictor.run_dir / "metrics.json")
    app = FastAPI(title=f"mlt: {info['name']}", version="1")

    class Records(BaseModel):
        records: list[dict]

    @app.get("/health")
    def health():
        return {"status": "ok", "run": info["name"], "task": predictor.task, "metric": info["metric"],
                "cv": info["cv"], "folds": len(predictor.artifacts)}

    @app.post("/predict")
    def predict(body: Records):
        if not body.records:
            raise HTTPException(400, "records is empty")
        start = time.perf_counter()
        try:
            pred = predictor.predict(pd.DataFrame(body.records))
        except (KeyError, ValueError) as exc:
            raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc
        return {"predictions": predictor.labels(pred), "scores": pred.tolist(),
                "columns": predictor.enc.pred_columns(), "time_ms": round(1000 * (time.perf_counter() - start), 1)}

    return app


def serve(run_dir: str, host: str = "127.0.0.1", port: int = 8000):
    uvicorn = require("uvicorn", "serve")
    uvicorn.run(create_app(run_dir), host=host, port=port)
