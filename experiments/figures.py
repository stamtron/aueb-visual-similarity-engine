"""Figures for the README and the lecture deck.

    python experiments/figures.py /path/to/images

Writes to lecture/assets/. Uses the cached embeddings from eval_search.py when present
and the clustering summary from a `vsim cluster` run (dataset id via --dataset).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from vsim.calibration import calibrate, loo_neighbour_sims, nn1_threshold, reference_quantiles
from vsim.images import list_images

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "lecture" / "assets"
CACHE = Path(__file__).parent / "cache"

# Reference palette (validated: scripts/validate_palette.js "#2a78d6,#eb6834" --mode light)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def font(size):
    for f in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def tile(path, size=220):
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(bg, im)
    return ImageOps.pad(im.convert("RGB"), (size, size), color=SURFACE)


def sheet(cells, cols, size=220, pad=10, caption_h=28, title=None):
    """cells: list of (path, caption, highlight: None|'good'|'bad')."""
    rows = (len(cells) + cols - 1) // cols
    top = 44 if title else 0
    W = cols * (size + pad) + pad
    H = top + rows * (size + caption_h + pad) + pad
    out = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(out)
    if title:
        d.text((pad, 10), title, fill=INK, font=font(22))
    for n, (path, cap, hl) in enumerate(cells):
        r, c = divmod(n, cols)
        x, y = pad + c * (size + pad), top + pad + r * (size + caption_h + pad)
        out.paste(tile(path, size), (x, y))
        if hl:
            d.rectangle([x - 3, y - 3, x + size + 2, y + size + 2],
                        outline=S1 if hl == "good" else S2, width=5)
        d.text((x + 2, y + size + 4), cap, fill=INK2, font=font(16))
    return out


def save(img_or_fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / name
    if isinstance(img_or_fig, Image.Image):
        img_or_fig.save(p, quality=88)
    else:
        img_or_fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(img_or_fig)
    print("wrote", p.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--dataset", help="clustering dataset dir (contains summary.json)")
    a = ap.parse_args()
    files = list_images(a.folder)
    by_id = {f.id: f.path for f in files}
    ids = [f.id for f in files]
    pos = {i: n for n, i in enumerate(ids)}

    # 1. What the data looks like
    rng = np.random.default_rng(3)
    pick = sorted(rng.choice(ids, 24, replace=False).tolist())
    save(sheet([(by_id[i], f"#{i}", None) for i in pick], 8, 180), "data_overview.jpg")

    # 2. Near-duplicates: same photo reused (pHash groups, the biggest first). Computed,
    #    not hand-picked, so the figure works on any folder.
    from eval_search import phash_groups  # same directory; imagehash is only needed here

    groups = [[ids[n] for n in g]
              for g in sorted(phash_groups(files), key=len, reverse=True)[:4]]
    for g in groups:
        save(sheet([(by_id[i], f"#{i}", None) for i in g], len(g), 240),
             f"neardup_{g[0]}.jpg")

    # 3. Query strips: the first image of a near-duplicate group + its top 5 per cached
    #    config. Its group mates are framed "good", anything else "bad".
    traps = {g[0]: set(g[1:]) for g in groups[:2]}
    for cache in sorted(CACHE.glob("*.npy")):
        emb = np.load(cache)
        if len(emb) != len(ids):
            continue
        sims = emb @ emb.T
        for q, mates in traps.items():
            order = [j for j in np.argsort(-sims[pos[q]]) if j != pos[q]][:5]
            cells = [(by_id[q], f"query #{q}", None)]
            cells += [(by_id[ids[j]], f"#{ids[j]}  cos {sims[pos[q], j]:.2f}",
                       "good" if ids[j] in mates else "bad") for j in order]
            save(sheet(cells, 6, 200), f"trap_{q}_{cache.stem}.jpg")

    # 4. Calibration curve (CLIP)
    clip = CACHE / "clip-b32.npy"
    if clip.exists():
        emb = np.load(clip)
        ref = reference_quantiles(emb)
        thr = nn1_threshold(emb)
        xs = np.linspace(0.35, 1.0, 400)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(xs, calibrate(xs, ref), color=S1, lw=2)
        ax.axvline(thr, color=INK2, lw=1, ls="--")
        ax.text(thr - 0.01, 0.92, f"outlier cut {thr:.2f}\n(p2 of nearest-neighbour cos)",
                ha="right", va="top", color=INK2, fontsize=10)
        points = [("typical top-1", float(np.median(loo_neighbour_sims(emb, 1))))]
        results = Path(__file__).parent / "results" / "search_eval.json"
        if results.exists():  # the noise probe's cosine, as measured by eval_search.py
            rows = [r for r in json.loads(results.read_text()) if r["model"] == "clip-b32"]
            if rows:
                points.insert(0, ("random noise", rows[0]["ood"]["random noise"]["top1_cos"]))
        for label, c in points:
            s = float(calibrate(c, ref))
            ax.plot([c], [s], "o", ms=8, color=S1, mec=SURFACE, mew=2)
            ax.annotate(f"{label}: cos {c:.2f} → {s:.2f}", (c, s),
                        xytext=(8, 10) if s < 0.5 else (10, -14),
                        textcoords="offset points", fontsize=10, color=INK)
        ax.set_xlabel("raw cosine similarity")
        ax.set_ylabel("score")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Score = share of this catalogue's neighbour pairs below the cosine")
        save(fig, "calibration.png")

        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        nn1 = loo_neighbour_sims(emb, 1)[:, 0]
        ax.hist(nn1, bins=40, color=S1, edgecolor=SURFACE, linewidth=1)
        ax.axvline(thr, color=INK2, lw=1, ls="--")
        ax.set_xlabel("cosine to nearest other image")
        ax.set_ylabel("images")
        ax.set_title("Every image's nearest neighbour (CLIP ViT-B/32)")
        save(fig, "nn1_hist.png")

    # 5. Clustering: K selection (two panels, one axis each) + cluster grids
    if a.dataset:
        s = json.loads((Path(a.dataset) / "summary.json").read_text())
        t = s["k_selection"]
        ks = [r["k"] for r in t]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, key, title in [(a1, "silhouette", "Silhouette (higher = tighter)"),
                               (a2, "ari", "Seed-to-seed agreement, ARI")]:
            vals = [r[key] for r in t]
            ax.plot(ks, vals, color=S1, lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
            best = ks.index(s["k"])
            ax.plot([ks[best]], [vals[best]], "o", ms=11, color=S2, mec=SURFACE, mew=2)
            ax.annotate(f"K={s['k']}: {vals[best]:.2f}", (ks[best], vals[best]),
                        xytext=(10, -4), textcoords="offset points", fontsize=10)
            ax.set_title(title)
            ax.set_xlabel("K")
            ax.set_xticks(ks)
        a2.axhline(0.8, color=INK2, lw=1, ls="--")
        a2.text(ks[-1], 0.81, "stability bar 0.8", ha="right", va="bottom", fontsize=10,
                color=INK2)
        save(fig, "k_selection.png")

        for c in s["clusters"]:
            members = c["members"][:12]
            save(sheet([(by_id[i], f"#{i}", None) for i in members], 6, 170,
                       title=f"[{c['cluster']}] {c['label']}  (n={c['size']})"),
                 f"cluster_{c['cluster']}.jpg")


if __name__ == "__main__":
    main()
