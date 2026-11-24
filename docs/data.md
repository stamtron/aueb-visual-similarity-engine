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

## Reference dataset: H&M Personalized Fashion Recommendations

The [H&M competition on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
ships a catalogue of 105,542 articles, each with one studio photo (about 1166×1750) and
clean labels.

| column | example | used for |
|---|---|---|
| `article_id` | `0588119001` | the file name, without the leading zero (`588119001.jpg`) |
| `product_code` | `0588119` | groups colour variants of one product |
| `product_type_name` | `Sweater` | the main ground truth for precision@K |
| `colour_group_name`, `perceived_colour_master_name` | `Black`, `Black` | precision@K by colour |
| `graphical_appearance_name` | `All over pattern` | print vs solid |
| `product_group_name`, `garment_group_name`, `index_group_name` | `Garment Upper body`, `Knitwear`, `Ladieswear` | coarser groupings |
| `prod_name` | `Tilly sweater` | free text |

**Why this dataset?**

- It is in the same domain as the take-home: catalogue photos of clothing.
- The photos are large enough to see a print, a texture or a logo.
- It has clean labels. The engine never sees them, but the evaluation uses them as ground
  truth: precision@K, meaning how many of a query's top K share its product type (or
  colour). This complements the label-free checks in [evaluation.md](evaluation.md).

**Known limitations, stated up front:**

- **You need a Kaggle account**, and the images may not be redistributed. They live in
  the gitignored `data/` and never in this repo, and figures made from them stay in the
  gitignored `lecture/assets/`.
- Almost all photos are studio shots on a plain background with **no overlay text**. The
  "CLIP reads text" failure mode from the original data does not occur naturally here.
  The evaluation reproduces it by stamping banners onto copies of images
  ([evaluation.md](evaluation.md)), which also gives known answers.
- The labels are imbalanced, and one product often comes in many colours. Hence the
  stratified sample, with one colour variant per product.

### Getting it

Every command below is the same on macOS, Linux and Windows (PowerShell), except
saving the token in step 1.

1. **Kaggle API token.** On kaggle.com, go to *Settings → API → Create New Token* and
   copy the token (it starts with `KGAT_`). Save it where the `kaggle` tool looks for it:

   macOS / Linux:

   ```bash
   mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
   (umask 077; printf '%s' 'KGAT_...your token...' > ~/.kaggle/access_token)
   ```

   Windows (PowerShell):

   ```powershell
   New-Item -ItemType Directory -Force "$HOME\.kaggle" | Out-Null
   Set-Content -NoNewline -Path "$HOME\.kaggle\access_token" -Value 'KGAT_...your token...'
   ```

   An older "legacy" token comes as a `kaggle.json` file instead: put it in the same
   `.kaggle` folder (on macOS / Linux, also run `chmod 600 ~/.kaggle/kaggle.json`).
   A token is a password: never commit it or paste it into a shared document.

2. **Accept the competition rules** (once, in the browser):
   <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules>.
   Without this, every download fails with `403 Forbidden`.

3. **Install the download extra and run the script:**

   ```bash
   uv sync --extra data
   uv run python scripts/download_hm.py
   ```

The script:

- downloads and unzips the catalogue (`articles.csv`, 36 MB) if it is not there yet,
  with no `unzip` tool needed
- keeps one random colour variant per product (`product_code`), so the sample is not 30
  colours of the same T-shirt
- keeps the `--types` most common product types (default 20) and samples `--per-class`
  articles from each (default 50), with a fixed `--seed`: 1000 images
- fetches only the sampled photos, one by one through the Kaggle API (about 15–30
  minutes), instead of the full 29 GB image archive
- checks every photo decodes, shrinks it to 512 px on its longest side, and writes it to
  `data/images/<article_id>.jpg` atomically (`.part` → rename)
- writes `data/labels.csv` at the end, for the images it wrote
- is resumable: if the run stops, run it again, and images already on disk are skipped

| flag | default | meaning |
|---|---|---|
| `--types` | 20 | how many of the most common product types to keep |
| `--per-class` | 50 | articles per product type |
| `--seed` | 0 | sampling seed |
| `--images-dir` | none | read from an unzipped copy of the full `images/` folder instead of the API |
| `--articles`, `--out`, `--labels` | `data/raw/hm/articles.csv`, `data/images`, `data/labels.csv` | paths |

The default sample covers: Bag, Blouse, Bra, Cardigan, Dress, Earring, Garment Set,
Hoodie, Jacket, Jumpsuit/Playsuit, Shirt, Shorts, Skirt, Sneakers, Sweater, T-shirt, Top,
Trousers, Underwear bottom, Vest top.

## Alternative: Fashion Product Images (small), no account needed

[`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
on Hugging Face is a mirror of Kaggle's `paramaggarwal/fashion-product-images-small`:
44,072 catalogue photos from an Indian e-commerce site, stored as two parquet files
(about 260 MB) with the JPEG bytes embedded. It downloads without an account, which
makes it the quickest way to try the engine.

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

The catch: the images are **small (60×80)**. The models upscale to 224 px, so fine
texture (patterns, logos) is partly lost. That is why H&M is the reference.

### Getting it

```bash
uvx --from huggingface_hub hf download ashraq/fashion-product-images-small --repo-type dataset --local-dir data/raw/fashion-small
uv sync --extra data
uv run python scripts/download_fashion.py      # 20 article types x 50 = 1000 images
```

The script reads the label columns only (parquet is columnar, so this is fast), samples
the same way as the H&M script, checks every image decodes, and writes the original JPEG
bytes. It runs in a few seconds.

Both scripts write the same layout (`data/images/` + `data/labels.csv`), so use one at a
time.

## Datasets we did not use

### The Kaggle "dress patterns" dataset?

We tried it first: 15.7k dresses with crowd-sourced pattern labels. Its CSV holds only
image URLs, and the S3 bucket they point to (`we-attributes`) no longer exists
(`NoSuchBucket`, checked November 2026). A dataset that is a list of links can rot; one
that ships its bytes cannot. That is worth checking before you build on any dataset.

## Other data

The engine was first built on a small, **private** set of fashion marketing images (a
take-home dataset). That data, its brief and any figures made from it are **not** in
this repository and must not be added (see `AGENTS.md`).
