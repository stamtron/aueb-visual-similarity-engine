"""Evaluation of search configurations on an image folder, with and without labels.

Label-free checks (weak ground truth built from the images themselves):

1. Near-duplicate recall: pHash (Hamming <= 14) groups the same photo reused with other
   overlay text / language / crop. For an image in a group of size g, are its g-1 mates
   in its top g-1 results?
2. Augmentation recall@1: query with a transformed copy of an image; is the original #1?
3. Banner trap: does the model match the *text* on an image instead of the product?
   We stamp the same "SALE -50%" banner on 50 products and add them to the gallery, then
   query with 50 *other* products wearing that banner. The right answer is the clean
   original of the query; a bannered stranger ranked above it means the text won.
   Building the trap synthetically gives known answers, and works on any catalogue.

With labels (labels.csv next to the folder, as the download scripts write it):

4. precision@K: the share of an image's top K that has its product type (or colour),
   macro-averaged over classes. The engine never sees these labels.

Plus out-of-domain checks and timing.

    HF_HUB_OFFLINE=1 python experiments/eval_search.py data/images [--labels data/labels.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

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


def near_dup_groups(files) -> list[list[int]]:
    """pHash groups, or [] if they cannot be reused photos.

    Real reuse gives small groups (one photo, a few banners). Giant groups mean pHash is
    chaining look-alike shots (e.g. every white-background studio photo), so the "mates"
    are not duplicates and any metric or figure built on them would be meaningless.
    """
    groups = phash_groups(files)
    if groups and max(map(len, groups)) > MAX_GROUP:
        print(f"pHash groups up to {max(map(len, groups))} images: no real near-duplicates "
              "here, skipping them")
        return []
    return groups


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
BANNER = "SALE -50%"
N_TRAP = 50
MAX_GROUP = 10  # larger pHash groups are chains of look-alikes, not reused photos
KS = (5, 10)
LABEL_KEYS = {"type": ("product_type_name", "articleType"),  # H&M, Fashion Product Images
              "colour": ("perceived_colour_master_name", "baseColour")}


def banner(img: Image.Image, text: str = BANNER) -> Image.Image:
    """A red sale strip across the lower fifth, like the ones on real catalogue images."""
    im = img.copy()
    w, h = im.size
    d = ImageDraw.Draw(im)
    y0 = int(h * 0.78)
    d.rectangle([0, y0, w, int(h * 0.96)], fill=(200, 20, 30))
    try:
        f = ImageFont.load_default(size=int(h * 0.12))
    except TypeError:  # Pillow < 10.1 has no sized default font
        f = ImageFont.load_default()
    d.text((w / 2, (y0 + h * 0.96) / 2), text, fill="white", font=f, anchor="mm")
    return im


def trap_split(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint indices: (queries, distractors). A distractor is never a query's own image."""
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:N_TRAP], idx[N_TRAP:2 * N_TRAP]


def trap_result(q: np.ndarray, emb: np.ndarray, q_idx: np.ndarray, d_emb: np.ndarray) -> dict:
    """For each bannered query, is its clean original ranked #1 among the clean gallery
    plus the bannered distractors? And how often does a bannered stranger beat it?"""
    gallery = np.vstack([emb, d_emb])
    sims = q @ gallery.T
    own = sims[np.arange(len(q)), q_idx]
    beaten_by_banner = (sims[:, len(emb):] > own[:, None]).any(1)
    return {"recall@1": round(float(np.mean(sims.argmax(1) == q_idx)), 3),
            "banner_wins": round(float(beaten_by_banner.mean()), 3)}


def read_labels(path: Path, ids: list) -> dict[str, list[str]]:
    """{'type': [...], 'colour': [...]} aligned with `ids`, from whichever columns exist."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    first = next(iter(rows[0]))  # article_id or id
    by_id = {str(int(r[first])) if r[first].isdigit() else r[first]: r for r in rows}
    out = {}
    for name, cols in LABEL_KEYS.items():
        col = next((c for c in cols if c in rows[0]), None)
        if col and all(str(i) in by_id for i in ids):
            out[name] = [by_id[str(i)][col] for i in ids]
    return out


def precision_at_k(order: np.ndarray, labels: list[str], k: int) -> float:
    """Macro-averaged over classes: each class counts the same, whatever its size."""
    lab = np.array(labels)
    hit = (lab[order[:, :k]] == lab[:, None]).mean(1)
    return float(np.mean([hit[lab == c].mean() for c in np.unique(lab)]))


def fingerprint(files) -> str:
    """Cache key for a folder: its IDs and file sizes and times. Keying by model alone
    silently reused embeddings of a different folder."""
    h = hashlib.sha256()
    for f in files:
        st = f.path.stat()
        h.update(f"{f.id}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:12]


def ood_images():
    rng = np.random.default_rng(0)
    noise = Image.fromarray((rng.random((400, 400, 3)) * 255).astype(np.uint8))
    blue = Image.new("RGB", (400, 400), (30, 60, 200))
    text = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(text).text((40, 120), "Quarterly revenue report 2026", fill="black",
                              font_size=36)
    return {"random noise": noise, "solid blue": blue, "black text on white": text}


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--labels", type=Path, default=None,
                   help="labels CSV (default: labels.csv next to the folder, if present)")
    a = p.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    files = list_images(a.folder)
    images = [load_image(f.path) for f in files]
    ids = [f.id for f in files]
    label_path = a.labels or a.folder.parent / "labels.csv"
    labels = read_labels(label_path, ids) if label_path.is_file() else {}
    print(f"{len(files)} images; labels: {sorted(labels) or 'none'}")
    fp = fingerprint(files)
    q_idx, d_idx = trap_split(len(files))
    trap_q = [banner(images[i]) for i in q_idx]
    trap_d = [banner(images[i]) for i in d_idx]
    groups = near_dup_groups(files)
    total_slots = sum(len(g) * (len(g) - 1) for g in groups)
    print(f"{len(groups)} pHash groups, {sum(map(len, groups))} images, {total_slots} mate slots")

    rng = np.random.default_rng(0)
    aug_idx = rng.choice(len(files), 50, replace=False)

    rows = []
    for name, model, mask in CONFIGS:
        key = f"{model}{'_mask' if mask else ''}_{fp}"
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

        tq = embed(trap_q, model, mask).emb
        td = embed(trap_d, model, mask).emb
        trap = trap_result(tq, emb, q_idx, td)

        prec = {f"{lab_name}@{k}": round(precision_at_k(order, lab, k), 3)
                for lab_name, lab in labels.items() for k in KS}

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
               "n_images": len(files),
               "near_dup_recall": round(found / total_slots, 3) if total_slots else None,
               "near_dup_found": found,
               "near_dup_slots": total_slots, "aug_recall@1": aug,
               "banner_trap": trap, "precision": prec,
               "nn1": {"min": round(float(nn1.min()), 3), "p2": round(thr, 3),
                       "median": round(float(np.median(nn1)), 3)},
               "ood": ood, "embed_s": build_s, "search_us": round(query_us, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "search_eval.json").write_text(json.dumps(rows, indent=2))

    pk = sorted(rows[0]["precision"]) if rows else []
    print("\n| config | " + " | ".join(f"P@{k}" for k in pk)
          + " | near-dup | aug R@1 | banner R@1 | banner wins | OOD flagged | embed |")
    print("|---|" + "---|" * (len(pk) + 6))
    for r in rows:
        aug, t = r["aug_recall@1"], r["banner_trap"]
        nd = f"{r['near_dup_found']}/{r['near_dup_slots']}" if r["near_dup_slots"] else "n/a"
        n_ood = sum(v["flagged"] for v in r["ood"].values())
        print(f"| {r['config']} | " + " | ".join(f"{r['precision'][k]:.2f}" for k in pk)
              + f" | {nd} | {np.mean(list(aug.values())):.2f} | {t['recall@1']:.2f} "
              f"| {t['banner_wins']:.2f} | {n_ood}/{len(r['ood'])} | {r['embed_s']}s |")


if __name__ == "__main__":
    main()
