"""
Interactive Terminal Chat Interface for GourAI V1.
Allows you to chat with the model in an ongoing conversational prompt loop.
"""
import sys
import time
from pathlib import Path
import torch

from src.config import GourAIConfig
from src.device import get_device, set_seed
from src.model.transformer import GourAILM
from src.model.streamer import stream_generate
from src.tokenizer import get_tokenizer
from src.train.checkpoint import load_checkpoint


def interactive_chat():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    set_seed(1337)
    device = get_device()
    tokenizer = get_tokenizer()

    # Load checkpoint
    ckpt_path = Path("checkpoints/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/latest_ckpt.pt")

    if ckpt_path.exists():
        checkpoint = load_checkpoint(str(ckpt_path), device=device)
        cfg = (
            GourAIConfig.from_dict(checkpoint["config"])
            if checkpoint.get("config")
            else GourAIConfig()
        )
        model = GourAILM(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_loss = checkpoint.get("best_val_loss", "N/A")
    else:
        cfg = GourAIConfig()
        model = GourAILM(cfg)
        val_loss = "Untrained"

    model.to(device)
    model.eval()

    print("\n" + "=" * 65)
    print("🍛 GourAI V1 Interactive Terminal Chat")
    print("=" * 65)
    print(f"Device        : {device}")
    print(f"Checkpoint    : {ckpt_path} (Val Loss: {val_loss})")
    print("Commands      : Type 'exit' or 'quit' to end chat.")
    print("=" * 65 + "\n")

    conversation_history = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("GourAI: Farewell!")
            break

        # Append to prompt format
        prompt = f"{conversation_history}\nUser: {user_input}\nGourAI: "
        
        # Keep prompt within context window
        if len(prompt) > cfg.block_size:
            prompt = f"User: {user_input}\nGourAI: "

        print("GourAI: ", end="", flush=True)

        reply_chars = []
        for char in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            device=device,
        ):
            print(char, end="", flush=True)
            reply_chars.append(char)
            # Stop if model outputs a new User prompt or double newline
            current_text = "".join(reply_chars)
            if "\nUser:" in current_text or "\n\n" in current_text:
                break

        print()  # newline
        reply = "".join(reply_chars).replace("\nUser:", "").strip()
        conversation_history += f"\nUser: {user_input}\nGourAI: {reply}"
        # Keep history compact
        if len(conversation_history) > 200:
            conversation_history = conversation_history[-200:]


if __name__ == "__main__":
    interactive_chat()
