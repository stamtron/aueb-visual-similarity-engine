"""Image I/O: folder listing with ID parsing, and robust loading from paths or URLs.

The dataset itself is clean (all RGB JPEG), but query images from the web are not:
RGBA PNGs, palette GIFs, CMYK JPEGs, EXIF-rotated phone photos, huge files, HTML
pages pretending to be images. Everything goes through `load_image`, so indexing
and querying share exactly one decoding path.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_SIDE = 1024  # images are shrunk to this before any model sees them
TIMEOUT = (5, 15)  # (connect, read) seconds

# Keep Pillow's decompression-bomb guard on, at a sane level (~ 90 MP).
Image.MAX_IMAGE_PIXELS = 90_000_000


class ImageLoadError(ValueError):
    """Raised when an image cannot be fetched or decoded."""


@dataclass(frozen=True)
class ImageFile:
    id: int | str
    path: Path


def parse_id(path: Path) -> int | str:
    """`15.jpg` -> 15. Non-numeric stems are kept as strings, never replaced by a position."""
    stem = path.stem
    return int(stem) if re.fullmatch(r"\d+", stem) else stem


def list_images(folder: str | Path) -> list[ImageFile]:
    """All image files in `folder`, sorted by parsed ID (numeric order: 1, 2, ..., 10)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise ValueError(f"No image files found in {folder}")
    items = [ImageFile(parse_id(p), p) for p in files]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in folder (e.g. 1.jpg and 1.png)")
    return sorted(items, key=lambda i: (isinstance(i.id, str), i.id))


def normalise(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """EXIF-rotate, convert any mode to RGB (alpha composited on white), cap the size."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background
    else:
        img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return img


def _decode(data: bytes, source: str) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # cheap JPEG downscale on decode
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageLoadError(f"Could not decode an image from {source}: {e}") from e
    return normalise(img)


def fetch_bytes(url: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """Download with timeouts, status check and a size cap."""
    if not re.match(r"^https?://", url):
        raise ImageLoadError(f"Not an http(s) URL: {url!r}")
    try:
        with requests.get(url, timeout=TIMEOUT, stream=True,
                          headers={"User-Agent": "vsim/0.1"}) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if ctype and not ctype.startswith(("image/", "application/octet-stream")):
                raise ImageLoadError(f"URL did not return an image (Content-Type: {ctype})")
            chunks, size = [], 0
            for chunk in r.iter_content(64 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ImageLoadError(f"Image larger than {max_bytes // 2**20} MB")
                chunks.append(chunk)
    except requests.RequestException as e:
        raise ImageLoadError(f"Could not download {url}: {e}") from e
    return b"".join(chunks)


def load_image(source: str | Path) -> Image.Image:
    """Load an RGB image from a local path or an http(s) URL."""
    s = str(source)
    if re.match(r"^https?://", s):
        return _decode(fetch_bytes(s), s)
    path = Path(s)
    if not path.is_file():
        raise ImageLoadError(f"Not a file or http(s) URL: {s!r}")
    return _decode(path.read_bytes(), s)


def load_image_bytes(source: str | Path) -> tuple[Image.Image, bytes]:
    """Like `load_image` but also returns the raw bytes (used for content hashing)."""
    s = str(source)
    data = fetch_bytes(s) if re.match(r"^https?://", s) else Path(s).read_bytes()
    return _decode(data, s), data
