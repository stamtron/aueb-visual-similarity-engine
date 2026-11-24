# Data

## The folder contract

The engine needs only a flat folder of images:

```
data/images/
  1163.jpg
  1164.jpg
  ...
```

- **Formats:** `.jpg .jpeg .png .webp .bmp .gif .tif .tiff`. Anything Pillow decodes is
  converted to RGB.
- **IDs:** the file name without its extension. Numeric names become `int`s and sort
  numerically; other names stay strings. Two files with the same ID (`1.jpg` and
  `1.png`) are an error.
- **Bad files:** unreadable files are skipped with a warning, and listed under
  `meta["skipped"]` in the index. They do not abort the build.
- **Subfolders:** ignored.

## Reference dataset: Fashion Product Images (small)

[`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
on Hugging Face is a mirror of Kaggle's `paramaggarwal/fashion-product-images-small`:
44,072 catalogue photos from an Indian e-commerce site, stored as two parquet files
(about 260 MB) with the JPEG bytes embedded.

| column | example | values |
|---|---|---|
| `id` | `1526` | unique product id → the file name |
| `articleType` | `Backpacks` | 141 (Tshirts 7065, Shirts 3212, Casual Shoes 2845, …) |
| `subCategory` | `Bags` | 45 |
| `masterCategory` | `Accessories` | 7 (Apparel, Accessories, Footwear, …) |
| `baseColour` | `Black` | 46 |
| `gender`, `usage`, `season` | `Unisex`, `Sports`, `Fall` | 5, 8, 4 |
| `productDisplayName` | `Puma Big Cat Backpack Black` | free text |
| `image` | 60×80 JPEG | |

**Why this dataset?**

- It is public and downloads without an account.
- It is in the same domain as the take-home (catalogue images of clothing and
  accessories).
- It has clean labels. The engine never sees them, but the evaluation uses them as ground
  truth: precision@K, meaning how many of a query's top-K share its article type (or
  colour). This complements the label-free checks in [evaluation.md](evaluation.md).

**Known limitations, stated up front:**

- The images are **small (60×80)**. The models upscale to 224 px, so fine texture
  (patterns, logos) is partly lost. The full-resolution Kaggle version is ~25 GB, too big
  for a laptop exercise.
- Almost all photos are studio shots on a white background with **no overlay text**. The
  "CLIP reads text" failure mode from the original data does not occur naturally here.
  The evaluation reproduces it by stamping banners onto copies of images
  ([evaluation.md](evaluation.md)), which also gives known answers.
- The labels are imbalanced (T-shirts alone are 16%), hence the stratified sample.

### Getting it

1. Download the parquet files (no token needed):

   ```bash
   uvx --from huggingface_hub hf download ashraq/fashion-product-images-small \
     --repo-type dataset --local-dir data/raw/fashion-small
   ```

2. Turn them into an image folder (needs `pyarrow`, in the `data` extra:
   `uv sync --extra data`):

   ```bash
   uv run python scripts/download_fashion.py      # 20 types x 50 = 1000 images
   ```

   The script:
   - reads the label columns only (parquet is columnar, so this is fast)
   - keeps the `--types` most common article types (default 20) and samples
     `--per-class` images from each (default 50), with a fixed `--seed`
   - checks every image decodes, then writes the original JPEG bytes to
     `data/images/<id>.jpg` atomically (`.part` → rename)
   - writes `data/labels.csv` for the images it wrote
   - is idempotent: existing files are skipped

   It runs in a few seconds. The default sample covers: Backpacks, Belts, Briefs, Casual
   Shoes, Flip Flops, Formal Shoes, Handbags, Heels, Jeans, Kurtas, Perfume and Body Mist,
   Sandals, Shirts, Socks, Sports Shoes, Sunglasses, Tops, Tshirts, Wallets, Watches.

### Why not the Kaggle "dress patterns" dataset?

We tried it first: 15.7k dresses with crowd-sourced pattern labels. Its CSV holds only
image URLs, and the S3 bucket they point to (`we-attributes`) no longer exists
(`NoSuchBucket`, checked November 2026). A dataset that is a list of links can rot; one
that ships its bytes cannot. That is worth checking before you build on any dataset.

## Other data

The engine was first built on a small, **private** set of fashion marketing images (a
take-home dataset). That data, its brief and any figures made from it are **not** in
this repository and must not be added (see `AGENTS.md`).
