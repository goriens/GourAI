"""
FastAPI Route Handlers for GourAI V1.
Implements:
- GET /api/info -> Model architecture and runtime diagnostic info
- POST /api/generate -> Synchronous full completion JSON
- GET/POST /api/generate/stream -> Real-time Server-Sent Events (SSE) token streamer
"""
import time
import json
import asyncio
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
import torch

from src.api.schemas import GenerateRequest, GenerateResponse, ModelInfoResponse
from src.model.generator import generate
from src.model.streamer import stream_generate

router = APIRouter(prefix="/api", tags=["GourAI"])


def get_state(request: Request):
    app = request.app
    if getattr(app.state, "model", None) is None:
        from src.api.main import ensure_model_loaded
        ensure_model_loaded(app)
    return app.state


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(request: Request):
    """
    Returns model metadata, hyperparameter configuration, parameter count, and execution device.
    """
    app_state = get_state(request)
    cfg = app_state.config
    model = app_state.model

    total_params = sum(p.numel() for p in model.parameters())

    return ModelInfoResponse(
        model_name="GourAI V1",
        parameters=total_params,
        parameter_string=f"{total_params/1e6:.2f} Million",
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        device=str(app_state.device),
        checkpoint_loaded=app_state.checkpoint_loaded,
        status="ready",
    )


@router.post("/reload")
async def reload_weights(request: Request):
    """
    Reloads latest saved model weights from disk into running application memory.
    """
    from pathlib import Path
    from src.train.checkpoint import load_checkpoint
    from src.config import GourAIConfig
    from src.model.transformer import GourAILM

    app = request.app
    device = getattr(app.state, "device", None) or torch.device("cpu")
    ckpt_path = Path("checkpoints/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/latest_ckpt.pt")

    if ckpt_path.exists():
        ckpt = load_checkpoint(str(ckpt_path), device=device)
        cfg = GourAIConfig.from_dict(ckpt["config"]) if ckpt.get("config") else GourAIConfig()
        model = GourAILM(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        app.state.model = model
        app.state.config = cfg
        app.state.checkpoint_loaded = True
        val_loss = ckpt.get("best_val_loss", "N/A")
        return {"status": "reloaded", "checkpoint": str(ckpt_path), "best_val_loss": val_loss}
    else:
        return {"status": "no_checkpoint_found"}


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(payload: GenerateRequest, request: Request):
    """
    Generates text synchronously and returns complete completion payload.
    """
    app_state = get_state(request)
    model = app_state.model
    tokenizer = app_state.tokenizer
    device = app_state.device

    t0 = time.time()

    prompt_ids = tokenizer.encode(payload.prompt)
    if not prompt_ids:
        prompt_ids = [0]

    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Autoregressive generation
    output_tensor = generate(
        model=model,
        idx=input_tensor,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        repetition_penalty=payload.repetition_penalty,
    )

    new_token_ids = output_tensor[0, input_tensor.size(1):].tolist()
    generated_text = tokenizer.decode(new_token_ids)
    total_text = payload.prompt + generated_text

    dt = time.time() - t0
    tok_count = len(new_token_ids)
    tok_per_sec = tok_count / dt if dt > 0 else 0

    return GenerateResponse(
        prompt=payload.prompt,
        generated_text=generated_text,
        total_text=total_text,
        tokens_generated=tok_count,
        time_taken_ms=round(dt * 1000, 2),
        tokens_per_sec=round(tok_per_sec, 2),
    )


@router.get("/generate/stream")
@router.post("/generate/stream")
async def generate_stream(
    request: Request,
    prompt: Optional[str] = Query("ROMEO:\n", description="Prompt string"),
    max_new_tokens: Optional[int] = Query(100, ge=1, le=500),
    temperature: Optional[float] = Query(0.8, ge=0.0, le=2.0),
    top_k: Optional[int] = Query(40, ge=0),
    top_p: Optional[float] = Query(0.9, ge=0.0, le=1.0),
    repetition_penalty: Optional[float] = Query(1.0, ge=0.5, le=2.0),
):
    """
    Streams generated tokens in real-time as Server-Sent Events (SSE).
    """
    # If request is JSON POST, override query params with body payload
    if request.method == "POST":
        try:
            body = await request.json()
            prompt = body.get("prompt", prompt)
            max_new_tokens = body.get("max_new_tokens", max_new_tokens)
            temperature = body.get("temperature", temperature)
            top_k = body.get("top_k", top_k)
            top_p = body.get("top_p", top_p)
            repetition_penalty = body.get("repetition_penalty", repetition_penalty)
        except Exception:
            pass

    app_state = get_state(request)
    model = app_state.model
    tokenizer = app_state.tokenizer
    device = app_state.device

    async def event_generator():
        t0 = time.time()
        count = 0

        # Yield each token as an SSE data frame
        for token_str in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        ):
            count += 1
            payload = json.dumps({"token": token_str, "done": False})
            yield f"data: {payload}\n\n"
            # Yield control back to event loop for true non-blocking streaming
            await asyncio.sleep(0)

        dt = time.time() - t0
        tok_speed = count / dt if dt > 0 else 0

        # Final end-of-stream event
        final_payload = json.dumps(
            {
                "token": "",
                "done": True,
                "total_tokens": count,
                "time_ms": round(dt * 1000, 2),
                "tokens_per_sec": round(tok_speed, 2),
            }
        )
        yield f"data: {final_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
