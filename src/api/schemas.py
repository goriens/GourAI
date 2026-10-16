"""
Pydantic Schemas for GourAI V1 FastAPI Server.
Defines strict type validation for generation requests and response payloads.
"""
from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        default="ROMEO:\n",
        description="Input text prompt to condition text generation",
        max_length=256,
    )
    max_new_tokens: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum new tokens to generate",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for greedy argmax, >1.0 for high creativity",
    )
    top_k: Optional[int] = Field(
        default=40,
        ge=0,
        description="Top-K truncation cutoff. 0 or None disables Top-K",
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-P (nucleus) sampling cumulative probability threshold",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Penalty multiplier for repeated tokens",
    )


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    total_text: str
    tokens_generated: int
    time_taken_ms: float
    tokens_per_sec: float


class ModelInfoResponse(BaseModel):
    model_name: str
    parameters: int
    parameter_string: str
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    device: str
    checkpoint_loaded: bool
    status: str
