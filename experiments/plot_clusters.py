"""Pictures of a clustering: what is in each cluster, and how the clusters sit together.

    uv run vsim cluster data/images                       # prints ds_...
    uv run python experiments/plot_clusters.py ds_... data/images

Writes two images to lecture/assets/clusters/ (gitignored, because they contain dataset
photos):

- cluster_mosaic.jpg  one row per cluster: its name, its size, and the members closest
                      to the cluster centre. "Closest to the centre" shows what the
                      cluster *is*; random members would show what it *contains*.
- cluster_map.jpg     every image placed on a 2D map of the embedding space (t-SNE),
                      snapped to a grid so thumbnails do not overlap. Thin lines mark
                      cluster borders, and each cluster is named where it sits.

Why no colours per cluster? K is often above 8, and more than 8 categorical colours can
no longer be told apart (or not at all with colour-blindness). The photos already carry
identity, so borders and direct labels do the job without colour.

t-SNE is only a picture. Distances between far-apart groups on the map mean little; the
clusters themselves come from k-means on the full embedding, not from this 2D view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE

from vsim.clustering import store_root
from vsim.images import list_images, load_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lecture" / "assets" / "clusters"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def font(size: int) -> ImageFont.FreeTypeFont:
    for f in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def thumb(path: Path, size: int) -> Image.Image:
    # same decoder as the engine (EXIF rotation, alpha flattened to white)
    return ImageOps.pad(load_image(path), (size, size), color="white")


def short(label: str) -> str:
    """'studio product photo: shirt & jeans' -> 'shirt & jeans' (the format part is
    the same for every cluster of a catalogue, so it is noise on a figure)."""
    return label.split(": ", 1)[-1]


def central(emb: np.ndarray, members: np.ndarray, n: int) -> np.ndarray:
    """Indices of the `n` members most similar to their (normalised) centroid."""
    c = emb[members].mean(0)
    c /= np.linalg.norm(c)
    return members[np.argsort(-(emb[members] @ c))[:n]]


# ----------------------------------------------------------------------------- mosaic
def mosaic(paths, emb, assign, labels, per_row=10, size=150, pad=8, label_w=300):
    k = len(labels)
    order = sorted(range(k), key=lambda c: -(assign == c).sum())  # biggest first
    W = label_w + per_row * (size + pad) + pad
    H = 70 + k * (size + pad) + pad
    out = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(out)
    d.text((pad + 4, 18), f"{k} clusters, {len(assign)} images (k-means on CLIP embeddings)",
           fill=INK, font=font(28))
    for r, c in enumerate(order):
        y = 70 + r * (size + pad)
        members = np.flatnonzero(assign == c)
        d.text((pad + 4, y + size / 2 - 26), short(labels[c]), fill=INK, font=font(22))
        d.text((pad + 4, y + size / 2 + 4), f"cluster {c} · {len(members)} images",
               fill=INK2, font=font(18))
        for j, i in enumerate(central(emb, members, per_row)):
            out.paste(thumb(paths[i], size), (label_w + j * (size + pad), y))
        d.line([(pad, y + size + pad / 2), (W - pad, y + size + pad / 2)], fill=GRID)
    return out


# ----------------------------------------------------------------------------- map
def grid_layout(xy: np.ndarray, seed: int = 0) -> tuple[np.ndarray, int, int]:
    """Snap 2D points to distinct cells of a near-square grid, keeping neighbours close.

    Optimal assignment (Hungarian algorithm) of points to cells by squared distance.
    """
    n = len(xy)
    cols = int(np.ceil(np.sqrt(n * 1.3)))  # a little wider than tall
    rows = int(np.ceil(n / cols))
    xy = (xy - xy.min(0)) / np.ptp(xy, 0)
    gx, gy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    cells = np.c_[gx.ravel(), gy.ravel()]
    cost = ((xy[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    _, cell = linear_sum_assignment(cost)
    return np.c_[cell % cols, cell // cols], cols, rows


def cluster_map(paths, emb, assign, labels, size=56, seed=0):
    xy = TSNE(2, metric="cosine", init="pca", perplexity=30, random_state=seed).fit_transform(emb)
    pos, cols, rows = grid_layout(xy, seed)
    m = 24
    W, H = m * 2 + cols * size, 80 + m + rows * size
    out = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(out)
    d.text((m, 22), "The catalogue on one page: t-SNE of the embeddings, lines = cluster "
           "borders", fill=INK, font=font(28))
    top = 80
    cell = np.full((rows, cols), -1)
    for i, (cx, cy) in enumerate(pos):
        cell[cy, cx] = assign[i]
        out.paste(thumb(paths[i], size), (m + cx * size, top + cy * size))

    # borders: draw an edge wherever two neighbouring cells belong to different clusters
    for y in range(rows):
        for x in range(cols):
            a = cell[y, x]
            px, py = m + x * size, top + y * size
            if x + 1 < cols and a != cell[y, x + 1] and -1 not in (a, cell[y, x + 1]):
                d.line([(px + size, py), (px + size, py + size)], fill=INK, width=3)
            if y + 1 < rows and a != cell[y + 1, x] and -1 not in (a, cell[y + 1, x]):
                d.line([(px, py + size), (px + size, py + size)], fill=INK, width=3)

    # names: a label chip at the cluster's densest cell (most same-cluster cells within 2
    # steps). The mean position would fall in a gap when a cluster is split on the map.
    f = font(20)
    for c in range(len(labels)):
        p = pos[assign == c]
        dense = (np.abs(p[:, None] - p[None]).max(-1) <= 2).sum(1)
        at = p[np.argmax(dense)]
        text = f"{c} · {short(labels[c])}"
        cx, cy = m + at[0] * size + size / 2, top + at[1] * size + size / 2
        w = d.textlength(text, font=f)
        x0 = min(max(cx - w / 2 - 10, 4), W - w - 24)
        d.rounded_rectangle([x0, cy - 16, x0 + w + 20, cy + 16], radius=8, fill=SURFACE,
                            outline=INK, width=2)
        d.text((x0 + 10, cy - 12), text, fill=INK, font=f)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset id printed by `vsim cluster`")
    p.add_argument("folder", type=Path, help="the image folder that was clustered")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    d = store_root() / a.dataset
    ds = np.load(d / "clusters.npz", allow_pickle=False)
    summary = json.loads((d / "summary.json").read_text())
    by_id = {str(f.id): f.path for f in list_images(a.folder)}
    paths = [by_id[i] for i in ds["ids"]]
    emb, assign, labels = ds["emb"], ds["assign"], [str(x) for x in ds["labels"]]
    print(f"{a.dataset}: {len(paths)} images, K={summary['k']}")

    a.out.mkdir(parents=True, exist_ok=True)
    for name, img in [("cluster_mosaic.jpg", mosaic(paths, emb, assign, labels)),
                      ("cluster_map.jpg", cluster_map(paths, emb, assign, labels, seed=a.seed))]:
        img.save(a.out / name, quality=90)
        print(f"  -> {a.out / name}  ({img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
