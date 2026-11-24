"""Turn the Hugging Face "fashion product images (small)" parquet files into an image folder.

The dataset (ashraq/fashion-product-images-small, a mirror of Kaggle's
paramaggarwal/fashion-product-images-small) has 44k catalogue photos with clean labels:
articleType (141 values), baseColour, gender, usage. The images are small (60x80 JPEG).

Download the parquet files once (about 260 MB):

    uvx --from huggingface_hub hf download ashraq/fashion-product-images-small \\
        --repo-type dataset --local-dir data/raw/fashion-small

Then this script writes what `vsim index` expects:

    data/images/<id>.jpg      one file per product, ID = the dataset's product id
    data/labels.csv           id, articleType, subCategory, masterCategory, baseColour, ...

The labels are not used by the engine. They are only for evaluation (precision@K by
article type and by colour). The sample is stratified: the most common article types,
the same number of images each, so T-shirts (16% of the data) do not dominate.

Examples
    python scripts/download_fashion.py                       # 20 types x 50 = 1000 images
    python scripts/download_fashion.py --types 30 --per-class 100
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from vsim.images import ImageLoadError, _decode

LABEL_COLS = ["articleType", "subCategory", "masterCategory", "baseColour", "gender",
              "usage", "season", "productDisplayName"]


def _pq():
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # the core install stays small; pyarrow is in the 'data' extra
        raise SystemExit("This script needs pyarrow: uv sync --extra data") from e
    return pq


def read_labels(files: list[Path]) -> list[dict]:
    """All rows without the image column (fast: parquet is columnar)."""
    pq = _pq()
    rows = []
    for f in files:
        rows.extend(pq.read_table(f, columns=["id", *LABEL_COLS]).to_pylist())
    return rows


def stratified(rows: list[dict], n_types: int, per_class: int, seed: int) -> list[dict]:
    """`per_class` random rows from each of the `n_types` most common article types."""
    counts = Counter(r["articleType"] for r in rows)
    keep = [t for t, n in counts.most_common() if n >= per_class][:n_types]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["articleType"] in keep:
            by_type[r["articleType"]].append(r)
    rng = random.Random(seed)
    sample = []
    for t in sorted(by_type):
        items = sorted(by_type[t], key=lambda r: r["id"])
        sample.extend(rng.sample(items, per_class))
    return sample


def write_images(files: list[Path], wanted: set[int], out: Path) -> tuple[set[int], list[str]]:
    """Copy the JPEG bytes of the wanted ids into `out`. Returns (written ids, errors)."""
    pq = _pq()
    done, errors = set(), []
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["id", "image"])
            for pid, img in zip(t.column("id").to_pylist(), t.column("image").to_pylist()):
                if pid not in wanted:
                    continue
                dest = out / f"{pid}.jpg"
                if not dest.exists():
                    data = img["bytes"]
                    try:
                        _decode(data, f"id {pid}")  # reject corrupt bytes before writing
                    except ImageLoadError as e:
                        errors.append(str(e))
                        continue
                    tmp = dest.with_suffix(".part")
                    tmp.write_bytes(data)  # already JPEG: keep the original bytes
                    tmp.replace(dest)
                done.add(pid)
    return done, errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("data/raw/fashion-small"),
                   help="folder from `hf download` (default data/raw/fashion-small)")
    p.add_argument("--out", type=Path, default=Path("data/images"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--types", type=int, default=20, help="most common article types (default 20)")
    p.add_argument("--per-class", type=int, default=50, help="images per type (default 50)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    files = sorted(a.src.glob("**/*.parquet"))
    if not files:
        print(f"No parquet files under {a.src}. Download them first (see README, "
              "'Get the data').", file=sys.stderr)
        return 2

    rows = read_labels(files)
    sample = stratified(rows, a.types, a.per_class, a.seed)
    print(f"{len(rows)} products; sampling {len(sample)} "
          f"({a.per_class} x {len(sample) // a.per_class} article types) into {a.out}/")

    a.out.mkdir(parents=True, exist_ok=True)
    done, errors = write_images(files, {r["id"] for r in sample}, a.out)

    good = sorted((r for r in sample if r["id"] in done), key=lambda r: r["id"])
    a.labels.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.labels.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", *LABEL_COLS])
        w.writeheader()
        w.writerows(good)
    tmp.replace(a.labels)

    print("per type: " + ", ".join(f"{t}={n}" for t, n in
                                   sorted(Counter(r["articleType"] for r in good).items())))
    if errors:
        print(f"{len(errors)} unreadable images skipped, e.g. {errors[0]}")
    print(f"{len(good)} images -> {a.out}/, labels -> {a.labels}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
