import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import orchestrator

app = FastAPI(title="BizCopilot")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    result = orchestrator.answer(req.question)
    return {
        "memo": result["memo"],
        "grounded": result["grounded"],
        "unverified_numbers": result["unverified_numbers"],
        "latency_sec": result["latency_sec"],
        "plan": result["plan"],
    }


@app.get("/")
def root():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
