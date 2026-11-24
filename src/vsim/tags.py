"""Zero-shot attribute tags with a CLIP text encoder, used for two things:

* explanations in search ("both: studio product photo, dress, black"), computed as the
  *intersection* of the query's and the result's tags, so every result gets its own
  reason and nothing is claimed that only one of the two images has;
* automatic, human-readable cluster labels.

Tags are probabilities from a per-category softmax over prompts. They are computed on
the original (unmasked) image, because "sale banner with text" is itself a useful tag.
"""

from __future__ import annotations

import functools

import numpy as np

VOCAB: dict[str, list[str]] = {
    "format": [
        "studio product photo", "model posing outdoors", "runway show", "sale banner",
        "brand logo banner", "close-up of a product", "group of models", "lifestyle photo",
    ],
    # Broad on purpose: one vocabulary for whole catalogues (clothes, shoes, accessories),
    # so a cluster of socks is not forced to be called "boots" for lack of a better word.
    "garment": [
        "dress", "shirt", "t-shirt", "top", "blouse", "tunic", "jeans", "trousers", "leggings",
        "skirt", "shorts", "jacket", "coat", "sweater", "cardigan", "hoodie", "suit",
        "swimwear", "bra", "underwear briefs", "socks", "sneakers", "formal leather shoes",
        "high heels", "sandals", "flip flops", "boots", "handbag", "backpack", "wallet",
        "belt", "scarf", "hat", "sunglasses", "jewellery", "watch", "perfume bottle",
    ],
    "colour": [
        "black", "white", "grey", "red", "pink", "blue", "navy", "green", "yellow", "orange",
        "brown", "beige", "purple", "colourful",
    ],
    "pattern": ["plain", "striped", "floral", "checked", "denim", "leather", "animal print",
                "graphic print", "knitted", "sequined"],
}

TEMPLATES = {
    "format": ["a {}", "a fashion {}"],
    "garment": ["a photo of a {}", "a fashion photo featuring a {}"],
    "colour": ["a photo of {} clothing", "a fashion image in {}"],
    "pattern": ["a photo of {} clothing", "a {} fabric"],
}

MIN_PROB = 0.15  # a tag "applies" to an image if it is in the top-2 of its category and >= this
TOP_PER_CATEGORY = 2

TAG_NAMES: list[str] = [f"{c}:{t}" for c, ts in VOCAB.items() for t in ts]


@functools.lru_cache(maxsize=4)
def _text_bank(tagger) -> np.ndarray:
    """(n_tags, dim) text embeddings, prompt-ensembled per tag."""
    rows = []
    for cat, terms in VOCAB.items():
        for term in terms:
            e = tagger.embed_texts([t.format(term) for t in TEMPLATES[cat]]).mean(0)
            rows.append(e / np.linalg.norm(e))
    return np.stack(rows).astype(np.float32)


def tag_probs(tagger, image_emb: np.ndarray) -> np.ndarray:
    """(n_images, n_tags) per-category softmax probabilities from tagger-space embeddings."""
    logits = tagger.logit_scale * (np.atleast_2d(image_emb) @ _text_bank(tagger).T)
    out = np.empty_like(logits)
    start = 0
    for terms in VOCAB.values():
        sl = slice(start, start + len(terms))
        z = logits[:, sl] - logits[:, sl].max(1, keepdims=True)
        e = np.exp(z)
        out[:, sl] = e / e.sum(1, keepdims=True)
        start += len(terms)
    return out


def _slices() -> dict[str, slice]:
    out, start = {}, 0
    for cat, terms in VOCAB.items():
        out[cat] = slice(start, start + len(terms))
        start += len(terms)
    return out


def active_tags(probs: np.ndarray) -> dict[str, list[str]]:
    """The tags that apply to one image, per category, most likely first."""
    out = {}
    for cat, sl in _slices().items():
        p = probs[sl]
        order = np.argsort(-p)[:TOP_PER_CATEGORY]
        out[cat] = [VOCAB[cat][i] for i in order if p[i] >= MIN_PROB]
    return out


def top_tag(probs: np.ndarray, category: str) -> str:
    sl = _slices()[category]
    return VOCAB[category][int(np.argmax(probs[sl]))]


def explain(q_probs: np.ndarray, r_probs: np.ndarray, cos: float, score: float,
            near_dup_cos: float) -> str:
    """One-line reason for a single (query, result) pair."""
    if cos >= 0.999:
        return f"identical image (cosine {cos:.3f})"
    q, r = active_tags(q_probs), active_tags(r_probs)
    shared = [t for cat in ("format", "garment", "colour", "pattern")
              for t in q[cat] if t in r[cat]]
    parts = []
    if cos >= near_dup_cos:
        parts.append(f"near-duplicate (cosine {cos:.3f})")
    if shared:
        parts.append(("only shares: " if score < 0.5 else "both: ") + ", ".join(shared))
    else:
        parts.append("overall visual similarity, no shared attribute tags")
    qf, rf = top_tag(q_probs, "format"), top_tag(r_probs, "format")
    if qf != rf and cos < near_dup_cos:
        parts.append(f"different format ({qf} vs {rf})")
    return "; ".join(parts)
