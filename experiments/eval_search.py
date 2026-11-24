"""Label-free evaluation of search configurations on an image folder.

There are no relevance labels, so we build three kinds of weak ground truth:

1. Near-duplicate recall: pHash (Hamming <= 14) groups the same photo reused with other
   overlay text / language / crop. For an image in a group of size g, are its g-1 mates
   in its top g-1 results?
2. Augmentation recall@1: query with a transformed copy of an image; is the original #1?
Plus out-of-domain checks and timing.

    HF_HUB_OFFLINE=1 python experiments/eval_search.py /path/to/images
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from vsim.calibration import calibrate, loo_neighbour_sims, nn1_threshold, reference_quantiles
from vsim.embedding import embed
from vsim.images import list_images, load_image

CONFIGS = [
    ("CLIP ViT-B/32", "clip-b32", False),
    ("CLIP + text masking", "clip-b32", True),
    ("SigLIP2 B/16", "siglip2-b16", False),
    ("DINOv2 S/14", "dinov2-s14", False),
    ("DINOv2 + text masking", "dinov2-s14", True),
]
OUT = Path(__file__).parent / "results"
CACHE = Path(__file__).parent / "cache"


def phash_groups(files, max_dist=14):
    hashes = [imagehash.phash(Image.open(f.path)) for f in files]
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if hashes[i] - hashes[j] <= max_dist:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(files)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def augment(img: Image.Image, kind: str) -> Image.Image:
    if kind == "greyscale":
        return ImageOps.grayscale(img).convert("RGB")
    if kind == "downscale 128px":
        im = img.copy()
        im.thumbnail((128, 128))
        return im
    if kind == "JPEG q10":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=10)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "mirror":
        return ImageOps.mirror(img)
    if kind == "centre crop 70%":
        w, h = img.size
        return img.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
    raise ValueError(kind)


AUGS = ["greyscale", "downscale 128px", "JPEG q10", "mirror", "centre crop 70%"]


def ood_images():
    rng = np.random.default_rng(0)
    noise = Image.fromarray((rng.random((400, 400, 3)) * 255).astype(np.uint8))
    blue = Image.new("RGB", (400, 400), (30, 60, 200))
    text = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(text).text((40, 120), "Quarterly revenue report 2026", fill="black",
                              font_size=36)
    return {"random noise": noise, "solid blue": blue, "black text on white": text}


def main(folder: str):
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    files = list_images(folder)
    images = [load_image(f.path) for f in files]
    groups = phash_groups(files)
    total_slots = sum(len(g) * (len(g) - 1) for g in groups)
    print(f"{len(groups)} pHash groups, {sum(map(len, groups))} images, {total_slots} mate slots")

    rng = np.random.default_rng(0)
    aug_idx = rng.choice(len(files), 50, replace=False)

    rows = []
    for name, model, mask in CONFIGS:
        key = f"{model}{'_mask' if mask else ''}"
        cache = CACHE / f"{key}.npy"
        t0 = time.perf_counter()
        if cache.exists():
            emb = np.load(cache)
            build_s = json.loads((CACHE / f"{key}.json").read_text())["build_s"]
        else:
            try:
                emb = embed(images, model, mask).emb
            except Exception as ex:  # noqa: BLE001 - e.g. a model whose deps are missing
                print(f"skipping {name}: {ex!r}", flush=True)
                continue
            build_s = round(time.perf_counter() - t0, 1)
            np.save(cache, emb)
            (CACHE / f"{key}.json").write_text(json.dumps({"build_s": build_s}))
        sims = emb @ emb.T
        np.fill_diagonal(sims, -np.inf)
        order = np.argsort(-sims, axis=1)

        found = 0
        for g in groups:
            for i in g:
                top = set(order[i, : len(g) - 1])
                found += len(top & (set(g) - {i}))

        aug = {}
        for kind in AUGS:
            q = embed([augment(images[i], kind) for i in aug_idx], model, mask).emb
            aug[kind] = float(np.mean(np.argmax(q @ emb.T, axis=1) == aug_idx))

        thr = nn1_threshold(emb)
        nn1 = loo_neighbour_sims(emb, 1)[:, 0]
        ref = reference_quantiles(emb)
        oods = ood_images()
        q = embed(list(oods.values()), model, mask).emb
        ood = {k: {"top1_cos": round(float(c), 3),
                   "score": round(float(calibrate(c, ref)), 3),
                   "flagged": bool(c < thr)}
               for k, c in zip(oods, (q @ emb.T).max(1))}

        t = time.perf_counter()
        for _ in range(1000):
            np.argpartition(-(emb @ emb[0]), 10)[:10]
        query_us = (time.perf_counter() - t) / 1000 * 1e6

        row = {"config": name, "model": model, "text_mask": mask,
               "near_dup_recall": round(found / total_slots, 3), "near_dup_found": found,
               "near_dup_slots": total_slots, "aug_recall@1": aug,
               "nn1": {"min": round(float(nn1.min()), 3), "p2": round(thr, 3),
                       "median": round(float(np.median(nn1)), 3)},
               "ood": ood, "embed_s": build_s, "search_us": round(query_us, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "search_eval.json").write_text(json.dumps(rows, indent=2))

    print("\n| config | near-dup recall | aug R@1 (mean) | mirror | crop | noise flagged | embed |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        a = r["aug_recall@1"]
        print(f"| {r['config']} | {r['near_dup_found']}/{r['near_dup_slots']} ({r['near_dup_recall']:.0%}) "
              f"| {np.mean(list(a.values())):.2f} | {a['mirror']:.2f} | {a['centre crop 70%']:.2f} "
              f"| {'yes' if r['ood']['random noise']['flagged'] else 'no'} | {r['embed_s']}s |")


if __name__ == "__main__":
    main(sys.argv[1])
