"""
FastAPI Server Entrypoint for GourAI V1.
Loads GourAILM model and Tokenizer into application state on startup,
configures CORS middleware, mounts REST/SSE routes, and serves static web assets.
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch

from src.config import GourAIConfig
from src.device import get_device
from src.model.transformer import GourAILM
from src.tokenizer import get_tokenizer
from src.train.checkpoint import load_checkpoint
from src.api.routes import router


def ensure_model_loaded(app: FastAPI):
    """Ensures model and tokenizer are loaded in app.state."""
    if getattr(app.state, "model", None) is not None:
        return

    device = get_device()
    app.state.device = device
    app.state.tokenizer = get_tokenizer()

    ckpt_path = None
    for candidate in [
        Path("checkpoints/model.pt"),
        Path("checkpoints/best_model.pt"),
        Path("checkpoints/latest_ckpt.pt"),
    ]:
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path and ckpt_path.exists():
        ckpt = load_checkpoint(str(ckpt_path), device=device)
        if ckpt.get("config"):
            cfg = GourAIConfig.from_dict(ckpt["config"])
        else:
            cfg = GourAIConfig()
        model = GourAILM(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        app.state.checkpoint_loaded = True
    else:
        cfg = GourAIConfig()
        model = GourAILM(cfg)
        app.state.checkpoint_loaded = False

    model.to(device)
    model.eval()
    app.state.model = model
    app.state.config = cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Runs once on server startup and shutdown.
    Caches model weights and tokenizer in memory for zero-overhead inference requests.
    """
    print("\n[SERVER] Initializing GourAI V1 Server...")
    ensure_model_loaded(app)
    model = app.state.model
    print(f"[SERVER] GourAI Ready on device '{app.state.device}'! (Params: {sum(p.numel() for p in model.parameters()):,})")
    yield
    print("[SERVER] Shutting down GourAI V1 server...")


app = FastAPI(
    title="GourAI V1 API",
    description="High-performance Inference and Streaming Server for GourAI GPT Language Model",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Web UI and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST & SSE API routes
app.include_router(router)

# Mount web directory if it exists and has files
web_dir = Path("web")
if web_dir.exists() and (web_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory="web", html=True), name="static")
else:
    @app.get("/")
    def root():
        return {
            "app": "GourAI V1",
            "version": "1.0.0",
            "docs_url": "/docs",
            "api_info": "/api/info",
            "generate_url": "/api/generate",
            "stream_url": "/api/generate/stream",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
