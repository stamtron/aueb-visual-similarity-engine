"""Optional overlay-text removal: detect text with EasyOCR, then inpaint it away.

Why: catalogue and marketing images often carry overlay text. CLIP reads rendered text very well,
so two unrelated banners that both say "SALE -50%" end up as nearest neighbours (the
"overlay-text trap"). Masking text before embedding pushes the model towards the
garments instead. It costs roughly 0.3-1 s per image on CPU, so it is opt-in
(`text_mask=True`) and evaluated in `experiments/`.
"""

from __future__ import annotations

import functools

import numpy as np
from PIL import Image

DETECT_SIDE = 768  # run the detector on a downscaled copy, then scale the boxes back


@functools.lru_cache(maxsize=1)
def _reader():
    try:
        import easyocr
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("Text masking needs the 'ocr' extra: pip install 'aueb-visual-similarity-engine[ocr]'") from e
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def text_mask(img: Image.Image, pad: int = 6) -> np.ndarray:
    """Binary uint8 mask (H, W) of detected text regions in `img`."""
    import cv2

    w, h = img.size
    scale = min(1.0, DETECT_SIDE / max(w, h))
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else img
    horizontal, free = _reader().detect(np.asarray(small))
    mask = np.zeros((h, w), np.uint8)
    for x0, x1, y0, y1 in horizontal[0]:
        p0 = (int(x0 / scale) - pad, int(y0 / scale) - pad)
        p1 = (int(x1 / scale) + pad, int(y1 / scale) + pad)
        cv2.rectangle(mask, p0, p1, 255, thickness=-1)
    for poly in free[0]:
        pts = (np.asarray(poly, np.float32) / scale).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        cv2.polylines(mask, [pts], True, 255, thickness=2 * pad)
    return mask


def remove_text(img: Image.Image) -> tuple[Image.Image, float]:
    """Inpaint detected text. Returns the cleaned image and the masked area fraction."""
    import cv2

    mask = text_mask(img)
    frac = float((mask > 0).mean())
    if frac == 0:
        return img, 0.0
    out = cv2.inpaint(np.asarray(img), mask, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(out), frac
