import hashlib
import logging
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer
from cachetools import LRUCache

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class EmbeddingModel:

    _instance = None

    def __new__(cls, model_name):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name):

        if getattr(self, "initialized", False):
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model.eval()

        # OPTIMIZED SETTINGS
        self.max_length = 128  # lower = faster
        self.cache = LRUCache(maxsize=8000)

        # warmup once
        self.embed_query("warmup")

        self.initialized = True
        logger.info("[EMBEDDINGS] READY ✔")

    def _key(self, texts):
        return hashlib.sha256("||".join(texts).encode()).hexdigest()

    def _mean_pool(self, hidden, mask):
        mask = mask.unsqueeze(-1).type_as(hidden)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _forward(self, texts):

        key = self._key(texts)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True)

            emb = self._mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        result = emb.cpu().numpy().astype(np.float32)

        self.cache[key] = result
        return result

    def embed_documents(self, texts, batch_size=32):  #smaller batch = less GPU spikes

        texts = [
            f"passage: {t.strip()}"
            for t in texts
            if isinstance(t, str) and t.strip()
        ]

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        out = []

        for i in range(0, len(texts), batch_size):
            out.append(self._forward(texts[i:i+batch_size]))

        return np.vstack(out)

    def embed_query(self, text):

        if not text:
            return np.zeros((384,), dtype=np.float32)

        return self._forward([f"query: {text.strip()}"])[0]