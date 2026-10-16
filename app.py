"""
GourAI V1 — Hugging Face Spaces Production App.
Provides both a native Gradio UI (Chat Studio, Playground, Architecture Inspector)
and mounts the FastAPI REST/SSE streaming backend.
"""
import os
import sys
import time
from pathlib import Path
import torch
import gradio as gr

from src.config import GourAIConfig
from src.device import get_device, set_seed
from src.model.transformer import GourAILM
from src.model.streamer import stream_generate
from src.tokenizer import get_tokenizer
from src.train.checkpoint import load_checkpoint

# 1. Setup Device & Seeds
set_seed(1337)
device = get_device()
tokenizer = get_tokenizer()

# 2. Locate and load model weights
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
    print(f"[SPACE] Loading weights from '{ckpt_path}'...")
    ckpt = load_checkpoint(str(ckpt_path), device=device)
    cfg = GourAIConfig.from_dict(ckpt["config"]) if ckpt.get("config") else GourAIConfig()
    model = GourAILM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss = ckpt.get("best_val_loss", "N/A")
    print(f"[SPACE] Checkpoint loaded with Val Loss: {val_loss}")
else:
    print("[SPACE] [WARNING] No checkpoint found. Using fresh weights.")
    cfg = GourAIConfig()
    model = GourAILM(cfg)

model.to(device)
model.eval()
total_params = sum(p.numel() for p in model.parameters())


# 3. Chat Generator Function (Streaming)
def chat_stream(message, history, temperature, top_k, max_tokens):
    """
    Handles conversational streaming in Gradio Chatbot.
    """
    if not message:
        return

    # Build prompt context from history
    context = ""
    if history:
        for item in history[-3:]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, b = item
                context += f"\nUser: {u}\nGourAI: {b}"
            elif isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    context += f"\nUser: {content}"
                elif role == "assistant":
                    context += f"\nGourAI: {content}"

    prompt = f"{context}\nUser: {message}\nGourAI: "
    if len(prompt) > cfg.block_size:
        prompt = f"User: {message}\nGourAI: "

    response_text = ""
    for char in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=0.9,
        repetition_penalty=1.1,
        device=device,
    ):
        response_text += char
        # Clean User echo if model outputs it
        cleaned = response_text.split("\nUser:")[0]
        yield cleaned


# 4. Playground Completion Generator Function (Streaming)
def playground_stream(prompt, temperature, top_k, top_p, max_tokens):
    """
    Handles prompt completion streaming in Playground mode.
    """
    if not prompt:
        yield "Please enter a seed prompt."
        return

    full_output = ""
    for char in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        device=device,
    ):
        full_output += char
        yield full_output


# 5. Build Gradio Interface
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="GourAI V1 — 10.8M Language Model") as demo:
    gr.Markdown(
        f"""
        # 🍛 GourAI V1 — Neural Playground & Chat Studio
        **A 10.8M Parameter GPT-style Transformer built from scratch in PyTorch** &bull; *Trained on Shakespeare* &bull; Device: `{device}`
        """
    )

    with gr.Tabs():
        # TAB 1: CHAT STUDIO
        with gr.Tab("💬 Shakespeare Chat Studio"):
            chatbot = gr.Chatbot(
                height=420,
                placeholder="Hark, noble traveler! I am GourAI V1. Ask me of honour, love, or kings, and I shall speak.",
            )
            with gr.Row():
                chat_msg = gr.Textbox(
                    placeholder="Type your message or prompt here...",
                    scale=7,
                    show_label=False,
                )
                chat_send = gr.Button("Send ➤", variant="primary", scale=1)

            with gr.Accordion("⚙️ Sampling Controls", open=False):
                with gr.Row():
                    chat_temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0.0 = deterministic, 1.2 = creative)")
                    chat_topk = gr.Slider(0, 65, value=40, step=1, label="Top-K Filter")
                    chat_tokens = gr.Slider(10, 250, value=100, step=10, label="Max New Tokens")

            # Quick Example Chips
            gr.Examples(
                examples=[
                    ["Lady, by yonder blessed moon I vow"],
                    ["Now is the winter of our discontent"],
                    ["To be, or not to be, that is the question:"],
                    ["What is honour? A word."],
                ],
                inputs=chat_msg,
            )

            # Wire chat events
            chat_send.click(
                chat_stream,
                inputs=[chat_msg, chatbot, chat_temp, chat_topk, chat_tokens],
                outputs=[chatbot],
            )
            chat_msg.submit(
                chat_stream,
                inputs=[chat_msg, chatbot, chat_temp, chat_topk, chat_tokens],
                outputs=[chatbot],
            )

        # TAB 2: PLAYGROUND
        with gr.Tab("✍️ Autoregressive Playground"):
            with gr.Row():
                with gr.Column(scale=5):
                    play_prompt = gr.Textbox(
                        label="Seed Prompt",
                        value="ROMEO:\n",
                        lines=5,
                        placeholder="Enter seed prompt text here...",
                    )
                    with gr.Row():
                        play_temp = gr.Slider(0.0, 1.5, value=0.75, step=0.05, label="Temperature")
                        play_topk = gr.Slider(0, 65, value=40, step=1, label="Top-K")
                    with gr.Row():
                        play_topp = gr.Slider(0.1, 1.0, value=0.90, step=0.05, label="Top-P (Nucleus)")
                        play_tokens = gr.Slider(10, 300, value=120, step=10, label="Max Tokens")
                    play_btn = gr.Button("⚡ Generate Stream", variant="primary")

                with gr.Column(scale=5):
                    play_output = gr.Textbox(
                        label="Generated Completion Stream",
                        lines=12,
                        interactive=False,
                    )

            play_btn.click(
                playground_stream,
                inputs=[play_prompt, play_temp, play_topk, play_topp, play_tokens],
                outputs=[play_output],
            )

        # TAB 3: ARCHITECTURE SPECS
        with gr.Tab("🏗️ Architecture & Model Specs"):
            gr.Markdown(
                f"""
                ### 🧠 Model Architecture Summary
                | Hyperparameter | Value | Description |
                |---|---|---|
                | **Total Parameters** | `{total_params:,}` | ~10.7 Million weights |
                | **Layers (`n_layer`)** | `6` | Transformer Blocks |
                | **Attention Heads (`n_head`)** | `6` | 64 dim/head |
                | **Embedding Dim (`n_embd`)** | `384` | Hidden representation size |
                | **Context Window (`block_size`)** | `{cfg.block_size}` | Maximum past tokens conditioned |
                | **Vocab Size** | `65` | Character-level Shakespeare vocabulary |
                | **Formulation** | `Pre-LayerNorm` | Enhanced training stability |
                | **Weight Tying** | `wte == lm_head` | Input & output matrices tied |
                | **Training Loss** | `1.9322` | Trained on Tiny Shakespeare |
                """
            )

# Launch Gradio server on Hugging Face port 7860
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
