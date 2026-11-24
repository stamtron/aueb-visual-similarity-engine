"""The one embedding pipeline shared by indexing, searching, clustering and assignment.

Train/serve skew is the classic bug in this kind of take-home: the index is built with
one preprocessing path and queries go through another. Here both call `embed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import tags as tagmod
from .models import Embedder, get_embedder, get_tagger


@dataclass
class Embedded:
    emb: np.ndarray  # (n, d) main-model embeddings, L2-normalised
    tag_probs: np.ndarray  # (n, n_tags)


def embed(images: list[Image.Image], model: str, text_mask: bool = False,
          batch_size: int = 32) -> Embedded:
    main: Embedder = get_embedder(model)
    tagger = get_tagger(main)
    if text_mask:
        from .textmask import remove_text

        model_inputs = [remove_text(im)[0] for im in images]
    else:
        model_inputs = images
    emb = main.embed_images(model_inputs, batch_size)
    # Tags always come from the unmasked image; reuse the main pass when it is the same thing.
    tag_emb = emb if (tagger is main and not text_mask) else tagger.embed_images(images, batch_size)
    return Embedded(emb, tagmod.tag_probs(tagger, tag_emb).astype(np.float32))
