"""
Automated End-to-End Verification Suite for GourAI V1.
Tests every major subsystem:
1. Tokenizer round-trip encode/decode
2. Model parameter count & shapes
3. Forward pass & Cross-Entropy loss
4. Autoregressive sampling & generation
5. FastAPI REST endpoints (/api/info, /api/generate)
6. Real-time SSE streaming latency
7. Static Web GUI asset delivery
"""
import time
import unittest
import torch
from fastapi.testclient import TestClient

from src.config import GourAIConfig
from src.tokenizer import get_tokenizer
from src.model.transformer import GourAILM
from src.model.generator import generate
from src.api.main import app, ensure_model_loaded


class TestGourAIE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = GourAIConfig(
            vocab_size=65,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
        )
        cls.model = GourAILM(cls.config)
        cls.tokenizer = get_tokenizer()
        cls.client = TestClient(app)
        ensure_model_loaded(app)

    def test_01_tokenizer_roundtrip(self):
        sample_text = "To be, or not to be: that is the question!"
        encoded = self.tokenizer.encode(sample_text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(sample_text, decoded, "Tokenizer encode -> decode did not preserve exact string!")

    def test_02_model_architecture(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 100_000, "Model parameter count unexpectedly low")
        # Test forward pass
        dummy_idx = torch.randint(0, 65, (2, 16))
        dummy_targets = torch.randint(0, 65, (2, 16))
        logits, loss = self.model(dummy_idx, dummy_targets)
        self.assertEqual(logits.shape, (2, 16, 65))
        self.assertIsNotNone(loss)
        self.assertLess(loss.item(), 5.0, "Initial cross-entropy loss is too high")

    def test_03_autoregressive_generation(self):
        prompt_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        out = generate(self.model, prompt_ids, max_new_tokens=10, temperature=0.8, top_k=20)
        self.assertEqual(out.shape, (1, 13))

    def test_04_fastapi_info_endpoint(self):
        res = self.client.get("/api/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_name"], "GourAI V1")
        self.assertEqual(data["status"], "ready")

    def test_05_fastapi_generate_endpoint(self):
        payload = {"prompt": "ROMEO:\n", "max_new_tokens": 10, "temperature": 0.8}
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("generated_text", data)
        self.assertEqual(data["tokens_generated"], 10)
        self.assertGreater(data["tokens_per_sec"], 0)

    def test_06_fastapi_stream_endpoint(self):
        t0 = time.time()
        res = self.client.get("/api/generate/stream?prompt=HAMLET:&max_new_tokens=5")
        latency_ms = (time.time() - t0) * 1000
        self.assertEqual(res.status_code, 200)
        self.assertIn("data: ", res.text)
        self.assertLess(latency_ms, 5000, "Streaming latency took longer than expected")

    def test_07_web_gui_assets(self):
        # Verify index.html, style.css, app.js
        res_html = self.client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("GourAI", res_html.text)

        res_css = self.client.get("/style.css")
        self.assertEqual(res_css.status_code, 200)

        res_js = self.client.get("/app.js")
        self.assertEqual(res_js.status_code, 200)


if __name__ == "__main__":
    unittest.main()
