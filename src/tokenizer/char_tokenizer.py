"""
Character-level Tokenizer for GourAI V1.
Encodes text strings into integer IDs and decodes integer sequences back to text.
Uses checkpoints/vocab.json mapping.
"""
import json
from pathlib import Path
from typing import List, Union


class CharTokenizer:
    def __init__(self, vocab_path: str = "checkpoints/vocab.json"):
        self.vocab_path = Path(vocab_path)
        if self.vocab_path.exists():
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocab_size = data["vocab_size"]
            self.chars = data["chars"]
            self.stoi = data["stoi"]
            # JSON keys are always strings, so cast back to int
            self.itos = {int(k): v for k, v in data["itos"].items()}
        else:
            raise FileNotFoundError(
                f"Vocabulary file not found at {self.vocab_path}. Please run src.data.prepare first."
            )

    def encode(self, text: str) -> List[int]:
        """Convert a string of text into a list of integer token IDs."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: Union[List[int], List[List[int]]]) -> str:
        """Convert a list of integer token IDs back into a human-readable string."""
        if len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        return "".join([self.itos.get(int(i), "") for i in tokens])

    def __len__(self) -> int:
        return self.vocab_size


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    print("================ GourAI CharTokenizer Test ================")
    sample_text = "Hello GourAI, let's build GPT!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original Text : {sample_text}")
    print(f"Encoded Tokens: {encoded[:15]}... (Total: {len(encoded)} tokens)")
    print(f"Decoded Text  : {decoded}")
    print(f"Vocab Size    : {tokenizer.vocab_size}")
    assert sample_text == decoded, "Encoding and decoding mismatch!"
    print("[SUCCESS] CharTokenizer self-verification passed!")
    print("===========================================================")
