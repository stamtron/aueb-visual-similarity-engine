"""Assignment 1: visual search.

Function 1  `build_index(folder, index_path, ...)`  -> path of the written index
Function 2  `search(url, k, index_path)`            -> [(id, score, explanation), ...]

The index is a single `.npz` file (no pickle, so loading it cannot execute code):
ids, paths, embeddings, tag probabilities, the calibration reference distribution, the
out-of-distribution threshold and a JSON metadata string (model, options, versions).
Search is brute-force cosine: N x 512 floats (1 MB for ~500 images) and a query takes microseconds.
An ANN library (FAISS/HNSW) only pays off around 10^5-10^6 images.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import __version__
from . import tags as tagmod
from .calibration import calibrate, nn1_threshold, reference_quantiles
from .embedding import embed
from .images import ImageLoadError, list_images, load_image
from .models import DEFAULT_MODEL, resolve

log = logging.getLogger(__name__)

NEAR_DUP_COS = {"open_clip:ViT-B-32-quickgelu:openai": 0.95}
DEFAULT_NEAR_DUP = 0.95
REF_K = 10


class Result(NamedTuple):
    id: int | str
    score: float
    explanation: str


@dataclass
class Index:
    ids: list[int | str]
    paths: list[str]
    emb: np.ndarray
    tag_probs: np.ndarray
    ref_q: np.ndarray
    ood_threshold: float
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            ids=np.array([str(i) for i in self.ids]),
            id_is_int=np.array([isinstance(i, int) for i in self.ids]),
            paths=np.array(self.paths),
            emb=self.emb.astype(np.float32),
            tag_probs=self.tag_probs.astype(np.float32),
            ref_q=self.ref_q.astype(np.float32),
            ood_threshold=np.float32(self.ood_threshold),
            meta=np.array(json.dumps(self.meta)),
        )
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written index
        return path

    @classmethod
    def load(cls, path: str | Path) -> Index:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No index at {path}. Build one with build_index() first.")
        return _load_cached(str(path.resolve()), path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int) -> Index:  # mtime in the key: rebuilt index reloads
    with np.load(path, allow_pickle=False) as z:
        ids = [int(s) if is_int else str(s) for s, is_int in zip(z["ids"], z["id_is_int"])]
        meta = json.loads(str(z["meta"]))
        if meta.get("tag_names") != tagmod.TAG_NAMES:
            raise ValueError("Index was built with a different tag vocabulary; rebuild it.")
        return Index(ids, [str(p) for p in z["paths"]], z["emb"], z["tag_probs"], z["ref_q"],
                     float(z["ood_threshold"]), meta)


def build_index(folder: str | Path, index_path: str | Path, model: str = DEFAULT_MODEL,
                text_mask: bool = False, batch_size: int = 32) -> Path:
    """Function 1: embed every image in `folder` and write a searchable index."""
    t0 = time.perf_counter()
    files = list_images(folder)
    images, ok = [], []
    for f in files:
        try:
            images.append(load_image(f.path))
            ok.append(f)
        except ImageLoadError as e:  # one bad file should not kill the whole build
            log.warning("skipping %s: %s", f.path, e)
    if len(ok) < 2:
        raise ValueError("Need at least 2 readable images to build an index")
    e = embed(images, model, text_mask, batch_size)
    full = resolve(model)
    index = Index(
        ids=[f.id for f in ok],
        paths=[str(f.path) for f in ok],
        emb=e.emb,
        tag_probs=e.tag_probs,
        ref_q=reference_quantiles(e.emb, REF_K),
        ood_threshold=nn1_threshold(e.emb, 2.0),
        meta={
            "model": full,
            "text_mask": text_mask,
            "near_dup_cos": NEAR_DUP_COS.get(full, DEFAULT_NEAR_DUP),
            "n_images": len(ok),
            "skipped": [str(f.path) for f in files if f not in ok],
            "tag_names": tagmod.TAG_NAMES,
            "package_version": __version__,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "build_seconds": round(time.perf_counter() - t0, 2),
        },
    )
    out = index.save(index_path)
    log.info("indexed %d images in %.1fs -> %s", len(ok), time.perf_counter() - t0, out)
    return out


def search(url: str, k: int, index_path: str | Path) -> list[Result]:
    """Function 2: the `k` most similar dataset images to the image at `url`.

    Returns (id, score, explanation) tuples, best first, with score in [0, 1].
    If the query does not look like anything in the catalogue, the explanations say so,
    but results are still returned: the caller asked for K, and decides what to show.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    index = Index.load(index_path)
    q = embed([load_image(url)], index.meta["model"], index.meta["text_mask"])
    return rank(index, q.emb[0], q.tag_probs[0], k)


def rank(index: Index, q_emb: np.ndarray, q_probs: np.ndarray, k: int) -> list[Result]:
    cos = index.emb @ q_emb
    k = min(k, len(cos))
    top = np.argpartition(-cos, k - 1)[:k]
    top = top[np.lexsort((np.arange(len(top)), -cos[top]))]  # by score, stable on ties
    scores = calibrate(cos[top], index.ref_q)
    out_of_domain = float(cos[top[0]]) < index.ood_threshold
    near_dup = index.meta.get("near_dup_cos", DEFAULT_NEAR_DUP)
    results = []
    for i, s in zip(top, scores):
        why = tagmod.explain(q_probs, index.tag_probs[i], float(cos[i]), float(s), near_dup)
        if out_of_domain:
            why = "low confidence: query unlike anything in the catalogue; " + why
        results.append(Result(index.ids[i], round(float(s), 4), why))
    return results
