"""
Tokenizer package for GourAI V1.
Provides unified access to tokenizers (CharTokenizer baseline and future subword BPE).
"""
from src.tokenizer.char_tokenizer import CharTokenizer


def get_tokenizer(tokenizer_type: str = "char", vocab_path: str = "checkpoints/vocab.json") -> CharTokenizer:
    """
    Factory function to retrieve the appropriate tokenizer instance.
    """
    if tokenizer_type == "char":
        return CharTokenizer(vocab_path=vocab_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Supported: 'char'")


__all__ = ["CharTokenizer", "get_tokenizer"]
