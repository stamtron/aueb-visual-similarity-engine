"""Tests run without any model download: a fake embedder stands in for CLIP.

It embeds an image as its normalised 8x8 colour thumbnail (so similar-looking images
really are close) and text as a seeded random vector (so zero-shot tags are stable).
"""

import hashlib

import numpy as np
import pytest
from PIL import Image

from vsim import models


class FakeEmbedder(models.Embedder):
    name = "fake"
    has_text = True
    logit_scale = 100.0

    def preprocess(self, im):
        import torch

        return torch.from_numpy(np.asarray(im.resize((8, 8)), np.float32).ravel() / 255 - 0.5)

    def _encode_image(self, batch):
        return batch

    def embed_texts(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=192).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


@pytest.fixture(autouse=True)
def fake_models(monkeypatch, tmp_path):
    fake = FakeEmbedder()
    monkeypatch.setattr(models, "get_embedder", lambda spec="fake": fake)
    monkeypatch.setattr("vsim.embedding.get_embedder", lambda spec="fake": fake)
    monkeypatch.setenv("VSIM_HOME", str(tmp_path / "home"))
    return fake


def _blob(rng, base, size=(64, 48)):
    arr = np.clip(base + rng.normal(0, 12, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dataset(tmp_path):
    """30 images in 3 colour families (red / green / blue), named 1.jpg .. 30.jpg."""
    rng = np.random.default_rng(0)
    folder = tmp_path / "imgs"
    folder.mkdir()
    bases = [np.array([200, 30, 30]), np.array([30, 190, 40]), np.array([40, 40, 210])]
    for i in range(1, 31):
        _blob(rng, bases[(i - 1) % 3]).save(folder / f"{i}.jpg", quality=95)
    return folder
