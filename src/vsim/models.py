"""Embedding models behind one small interface.

Two families are supported:

* `open_clip` models (CLIP, SigLIP2, ...): image *and* text encoders, so they can also
  do zero-shot tagging for explanations and cluster labels.
* DINOv2: a self-supervised, image-only model. It reads rendered text far less than CLIP,
  which matters on this dataset (see README, "the overlay-text trap").

Models are cached per process (`get_embedder`), so a search or an assignment pays the
load cost once, never per image or per result.
"""

from __future__ import annotations

import functools
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

# Short names -> full specs. Any "open_clip:<arch>:<pretrained>" or "hf-hub:<repo>" also works.
MODEL_ALIASES = {
    "clip-b32": "open_clip:ViT-B-32-quickgelu:openai",
    "clip-b32-laion": "open_clip:ViT-B-32:laion2b_s34b_b79k",
    "siglip2-b16": "hf-hub:timm/ViT-B-16-SigLIP2-naflex",
    "dinov2-s14": "dinov2:dinov2_vits14",
}
DEFAULT_MODEL = "clip-b32"
TAGGER_MODEL = "clip-b32"  # used for explanations / labels when the main model has no text side

# The original `openai/CLIP` package caches its weights here; reuse them if present.
OPENAI_CLIP_CACHE = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"


def resolve(spec: str) -> str:
    return MODEL_ALIASES.get(spec, spec)


def device() -> torch.device:
    return torch.device(os.environ.get("VSIM_DEVICE", "cpu"))


def _l2(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    return (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy().astype(np.float32)


class Embedder:
    """Maps PIL images (and, if supported, text) to L2-normalised float32 vectors."""

    name: str
    has_text: bool = False

    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = torch.stack([self.preprocess(im) for im in images[i : i + batch_size]])
                out.append(_l2(self._encode_image(batch.to(device()))))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no text encoder")

    @property
    def dim(self) -> int:
        return int(self.embed_images([Image.new("RGB", (64, 64))]).shape[1])


class OpenClipEmbedder(Embedder):
    has_text = True

    def __init__(self, spec: str):
        import open_clip

        self.name = spec
        if spec.startswith("hf-hub:"):
            model, _, preprocess = open_clip.create_model_and_transforms(spec)
            tokenizer = open_clip.get_tokenizer(spec)
        else:
            _, arch, pretrained = spec.split(":", 2)
            if pretrained == "openai" and OPENAI_CLIP_CACHE.exists() and arch.startswith("ViT-B-32"):
                # Build the architecture empty (open_clip warns "initialized randomly"; it
                # is not, the weights are loaded on the next lines), then load the
                # TorchScript checkpoint's state dict into it.
                logging.getLogger("open_clip").setLevel(logging.ERROR)
                model, _, preprocess = open_clip.create_model_and_transforms(arch)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    jit = torch.jit.load(str(OPENAI_CLIP_CACHE), map_location="cpu")
                sd = jit.state_dict()
                for k in ("input_resolution", "context_length", "vocab_size"):
                    sd.pop(k, None)
                model.load_state_dict({k: v.float() for k, v in sd.items()})
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(arch)
        self.model = model.eval().to(device())
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def _encode_image(self, batch):
        return self.model.encode_image(batch)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            return _l2(self.model.encode_text(self.tokenizer(texts).to(device())))

    @functools.cached_property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())


class DinoEmbedder(Embedder):
    def __init__(self, spec: str):
        from torchvision import transforms as T

        self.name = spec
        variant = spec.split(":", 1)[1]
        local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if local.exists():
            model = torch.hub.load(str(local), variant, source="local")
        else:
            model = torch.hub.load("facebookresearch/dinov2", variant, trust_repo=True)
        self.model = model.eval().to(device())
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _encode_image(self, batch):
        return self.model(batch)  # CLS token embedding


@functools.lru_cache(maxsize=4)
def get_embedder(spec: str = DEFAULT_MODEL) -> Embedder:
    full = resolve(spec)
    log.info("loading model %s", full)
    if full.startswith("dinov2:"):
        return DinoEmbedder(full)
    if full.startswith(("open_clip:", "hf-hub:")):
        return OpenClipEmbedder(full)
    raise ValueError(f"Unknown model spec {spec!r}. Known aliases: {sorted(MODEL_ALIASES)}")


def get_tagger(main: Embedder) -> OpenClipEmbedder:
    """The main model if it has a text side, otherwise a CLIP model for zero-shot tagging."""
    return main if main.has_text else get_embedder(TAGGER_MODEL)  # type: ignore[return-value]
