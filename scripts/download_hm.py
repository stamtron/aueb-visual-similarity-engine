"""Build an image folder from the H&M Personalized Fashion Recommendations data (Kaggle).

The competition ships 105k articles with clean catalogue labels (product type, colour,
graphical appearance, garment group) and one studio photo per article, about 1166x1750.
That is large enough to see a print, unlike 60x80 thumbnails.

The data is under the competition's rules: you must accept them on Kaggle, and the
images may not be redistributed. So they live in data/ (gitignored), never in the repo.

Two ways to get the images:

1. Per file through the Kaggle API (default). Only the sampled images are downloaded,
   about 1-2 s each. The catalogue (articles.csv) is fetched and unzipped first if it
   is not there yet, so this one command is all you need, on any OS:

       uv run python scripts/download_hm.py

2. From the full archive (about 29 GB), if you already have it unzipped:

       python scripts/download_hm.py --images-dir data/raw/hm/images

Writes what `vsim index` expects:

    data/images/<article_id>.jpg   longest side <= 512 px, ID without the leading zero
    data/labels.csv                article_id, product_type_name, colour_group_name, ...

The labels are not used by the engine, only for evaluation. The sample is stratified:
the most common product types, the same number of articles each, and one colour variant
per product (product_code), so the sample is not 30 colours of one T-shirt.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from vsim.images import ImageLoadError, _decode

COMPETITION = "h-and-m-personalized-fashion-recommendations"
LABEL_COLS = ["product_code", "prod_name", "product_type_name", "product_group_name",
              "graphical_appearance_name", "colour_group_name",
              "perceived_colour_master_name", "index_group_name", "garment_group_name"]
MAX_SIDE = 512


def read_articles(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{"article_id": r["article_id"], **{c: r[c] for c in LABEL_COLS}}
                for r in csv.DictReader(f)]


def stratified(rows: list[dict], n_types: int, per_class: int, seed: int) -> list[dict]:
    """`per_class` random articles from each of the `n_types` most common product types,
    at most one per product_code."""
    rng = random.Random(seed)
    by_product: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["article_id"]):
        by_product[r["product_code"]].append(r)
    one = [rng.choice(v) for _, v in sorted(by_product.items())]
    counts = Counter(r["product_type_name"] for r in one)
    keep = [t for t, n in counts.most_common() if n >= per_class][:n_types]
    sample = []
    for t in sorted(keep):
        sample.extend(rng.sample([r for r in one if r["product_type_name"] == t], per_class))
    return sample


def rel_path(article_id: str) -> str:
    """Where the competition keeps an article's photo: images/010/0108775015.jpg."""
    return f"images/{article_id[:3]}/{article_id}.jpg"


def take(folder: Path, name: str) -> bytes | None:
    """Read and remove a downloaded file. Where it lands differs between kaggle versions
    (flat, or under images/058/), and it may arrive zipped, so search for it."""
    found = sorted(p for p in folder.rglob(f"{name}*") if p.is_file())
    if not found:
        return None
    f = found[0]
    if f.suffix == ".zip":
        with zipfile.ZipFile(f) as zf:
            data = zf.read(zf.namelist()[0])
    else:
        data = f.read_bytes()
    f.unlink()
    return data


def unpack(folder: Path, name: str, dest: Path) -> None:
    data = take(folder, name)
    if data is None:
        raise SystemExit(f"Kaggle did not deliver {name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


class KaggleSource:
    """Fetch single files with the Kaggle API (needs ~/.kaggle/kaggle.json or env vars)."""

    def __init__(self) -> None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:  # the core install stays small; kaggle is in 'data'
            raise SystemExit("This script needs the kaggle package: uv sync --extra data") from e
        self.api = KaggleApi()
        self.api.authenticate()
        self.tmp = Path(tempfile.mkdtemp(prefix="vsim_hm_"))

    def fetch_articles(self, dest: Path) -> None:
        """Download articles.csv to `dest`. It arrives zipped; unzip it in Python, so
        no `unzip` command is needed (Windows has none)."""
        self.api.competition_download_file(COMPETITION, "articles.csv", path=self.tmp)
        unpack(self.tmp, "articles.csv", dest)

    def get(self, article_id: str) -> bytes | None:
        name = rel_path(article_id)
        try:
            self.api.competition_download_file(COMPETITION, name, path=self.tmp, quiet=True)
        except Exception as e:  # the API raises many types; only "not found" is a skip
            if "404" in str(e) or "Not Found" in str(e):
                return None  # some articles have no photo
            raise
        return self.take(article_id)

    def take(self, article_id: str) -> bytes | None:
        return take(self.tmp, f"{article_id}.jpg")

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class FolderSource:
    """Read from an unzipped copy of the competition's images/ folder."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, article_id: str) -> bytes | None:
        f = self.root / article_id[:3] / f"{article_id}.jpg"
        return f.read_bytes() if f.is_file() else None

    def close(self) -> None:
        pass


def write_image(data: bytes, dest: Path) -> None:
    """Validate, shrink to MAX_SIDE, and write atomically (temp file, then rename)."""
    img = _decode(data, dest.name)  # raises ImageLoadError on corrupt bytes
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    tmp = dest.with_suffix(".part")
    img.save(tmp, "JPEG", quality=90)
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--articles", type=Path, default=Path("data/raw/hm/articles.csv"))
    p.add_argument("--images-dir", type=Path, default=None,
                   help="unzipped images/ folder; without it, files come from the Kaggle API")
    p.add_argument("--out", type=Path, default=Path("data/images"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--types", type=int, default=20, help="most common product types (20)")
    p.add_argument("--per-class", type=int, default=50, help="articles per type (50)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    src = FolderSource(a.images_dir) if a.images_dir else KaggleSource()
    if not a.articles.is_file():
        zipped = a.articles.with_name(a.articles.name + ".zip")
        if zipped.is_file():  # downloaded by hand with the kaggle CLI
            unpack(zipped.parent, zipped.name, a.articles)
        elif isinstance(src, KaggleSource):
            print(f"downloading {a.articles.name} (36 MB) ...")
            src.fetch_articles(a.articles)
        else:
            print(f"{a.articles} not found (see README, 'Get the data').", file=sys.stderr)
            return 2
    rows = read_articles(a.articles)
    sample = stratified(rows, a.types, a.per_class, a.seed)
    print(f"{len(rows)} articles; sampling {len(sample)} "
          f"({a.per_class} x {len(sample) // max(a.per_class, 1)} product types) into {a.out}/")

    a.out.mkdir(parents=True, exist_ok=True)
    good, missing, errors = [], 0, []
    try:
        for n, r in enumerate(sample, 1):
            dest = a.out / f"{int(r['article_id'])}.jpg"  # same ID rule as the engine
            if not dest.exists():
                data = src.get(r["article_id"])
                if data is None:
                    missing += 1
                    continue
                try:
                    write_image(data, dest)
                except ImageLoadError as e:
                    errors.append(str(e))
                    continue
            good.append({**r, "article_id": int(r["article_id"])})
            if n % 100 == 0:
                print(f"  {n}/{len(sample)}")
    finally:
        src.close()

    good.sort(key=lambda r: r["article_id"])
    a.labels.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.labels.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["article_id", *LABEL_COLS])
        w.writeheader()
        w.writerows(good)
    tmp.replace(a.labels)

    if missing:
        print(f"{missing} articles have no photo and were skipped")
    if errors:
        print(f"{len(errors)} unreadable images skipped, e.g. {errors[0]}")
    print(f"{len(good)} images -> {a.out}/, labels -> {a.labels}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
