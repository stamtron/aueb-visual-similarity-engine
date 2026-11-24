# Visual Similarity Engine

Image retrieval and clustering over a folder of images, built as a worked example of a
data-science take-home for the AUEB MSc Data Science careers lecture
(`aueb_data_science_careers.html`).

Given a folder of images it can:

- **Search**: build an index once, then for any query image (local path or URL) return
  the K most similar images, each with a **score in [0, 1]** and a short **explanation**
  (`both: studio product photo, dress, floral`).
- **Cluster**: group the folder into automatically **labelled** clusters (choosing K
  itself), then **assign** new images to a cluster, or flag them as an **outlier**.

It runs on a laptop CPU with no GPU or vector database. Embeddings come from
CLIP ViT-B/32 (default) or DINOv2, and search is brute-force cosine similarity.

```
$ vsim search https://example.com/red-floral-dress.jpg -k 3 --index index.npz   # illustrative output
 85123  0.973  both: studio product photo, dress, red, floral
 85410  0.951  both: dress, red, floral; different format (studio product photo vs model posing outdoors)
 84007  0.902  both: dress, floral
```

---

## Contents

- [Quick start](#quick-start)
- [Get the data](#get-the-data)
- [Usage: command line](#usage-command-line)
- [Usage: Python API](#usage-python-api)
- [Configuration](#configuration)
- [How it works (one paragraph each)](#how-it-works)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Development](#development)
- [Limitations](#limitations)
- [Documentation](#documentation)

---

## Quick start

Requirements:

- Python 3.10 or newer (3.12 is pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- About 2 GB of disk for PyTorch and the model weights

```bash
git clone git@github.com:stamtron/aueb-visual-similarity-engine.git
cd aueb-visual-similarity-engine

uv sync                      # core: search + clustering
uv sync --all-extras         # + OCR text masking, evaluation/figures, pyarrow for the data script

uv run pytest -q             # 22 tests, no model download needed (they use a fake embedder)
```

Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[ocr,eval,data]"`.

The first real run downloads the model weights:

- CLIP ViT-B/32: about 350 MB, via open_clip
- DINOv2 S/14: about 90 MB, via torch.hub

Both are cached afterwards, and after that the engine works offline (`HF_HUB_OFFLINE=1`).

Try it on any folder of images:

```bash
uv run vsim index path/to/images --out index.npz
uv run vsim search path/to/query.jpg -k 5 --index index.npz
```

## Get the data

The engine works on **any folder of images**. File names become IDs: `123.jpg` gets ID
`123`, and non-numeric names stay as strings.

The reference dataset is the **H&M Personalized Fashion Recommendations** catalogue
(Kaggle): 105k articles, each with one studio photo (about 1166×1750) and clean labels:
product type, colour, graphical appearance (print), garment group. The labels are
**not** used by the engine. They give the evaluation real ground truth: a "Sweater"
query should return sweaters.

The data is under the competition's rules: you need a Kaggle account, you must accept
the rules, and the images may not be redistributed. That is why they live in the
gitignored `data/` and never in this repo.

### H&M (recommended)

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

   The script first downloads and unzips the catalogue (`articles.csv`, 36 MB), then a
   balanced sample of images: 20 product types × 50 = 1000 images.

   Only the sampled photos are fetched, one by one, through the Kaggle API (about
   15–30 minutes), instead of the full 29 GB image archive. Each photo is checked,
   shrunk to 512 px on its longest side and written as `data/images/<article_id>.jpg`
   (the leading zero is dropped: `0588119001` → `588119001.jpg`). The labels go to
   `data/labels.csv`, written at the end. If the run stops, run it again: images
   already on disk are skipped.

   | flag | default | meaning |
   |---|---|---|
   | `--types` | 20 | how many of the most common product types to keep |
   | `--per-class` | 50 | articles per product type |
   | `--seed` | 0 | sampling seed |
   | `--images-dir` | none | read from an unzipped copy of the full `images/` folder instead of the API |
   | `--articles`, `--out`, `--labels` | `data/raw/hm/articles.csv`, `data/images`, `data/labels.csv` | paths |

   The sample keeps one colour variant per product (`product_code`), so it is not 30
   colours of the same T-shirt.

### Fashion Product Images (no account needed)

A quicker alternative, public on Hugging Face: 44k catalogue photos with an article
type, colour, gender and usage. The images are small (60×80 px), too small to see a
print, which is why H&M is the reference.

```bash
uvx --from huggingface_hub hf download ashraq/fashion-product-images-small --repo-type dataset --local-dir data/raw/fashion-small
uv sync --extra data
uv run python scripts/download_fashion.py      # 20 article types x 50 = 1000 images
```

Both scripts write the same layout (`data/images/` + `data/labels.csv`), so use one at a
time. See [docs/data.md](docs/data.md) for what each dataset means for the evaluation,
and for why we dropped the Kaggle "dress patterns" dataset (its image bucket no longer
exists).

`data/` is gitignored. Never commit images, CSVs or built indexes.

## Usage: command line

Every command takes `-v` for progress logs.

```bash
# --- search ---
vsim index data/images --out index.npz [--model clip-b32|dinov2-s14|...] [--text-mask]
vsim search <path-or-url> -k 5 --index index.npz

# --- clustering ---
vsim cluster data/images [--k 6] [--model ...] [--text-mask]   # prints a dataset id, e.g. ds_3f2a...
vsim clusters ds_3f2a...                                         # labels, sizes, example members, K table
vsim assign <path-or-url> ds_3f2a... [--dry-run]                 # prints a label, or "outlier"
```

Notes:

- `search` reads the model and options stored in the index, so you never pass `--model`
  twice.
- `cluster` is cached by content: running it again on the same folder returns the same
  id instantly.
- `assign` persists each new image (unless you pass `--dry-run`), so later assignments
  can see earlier ones. Assigning the same image twice is a no-op.

## Usage: Python API

```python
from vsim import build_index, search, preprocess, assign

# search
build_index("data/images", "index.npz", model="clip-b32", text_mask=False)
for r in search("https://example.com/query.jpg", k=5, index_path="index.npz"):
    print(r.id, r.score, r.explanation)          # Result is a NamedTuple

# clustering
ds = preprocess("data/images")                   # -> "ds_..."
a = assign("query.jpg", ds)                      # -> Assignment
print(a.label, a.is_outlier, a.similarity, a.threshold, a.nearest_ids)
```

## Configuration

| setting | where | default | effect |
|---|---|---|---|
| model | `--model` / `model=` | `clip-b32` | `clip-b32`, `clip-b32-laion`, `siglip2-b16`, `dinov2-s14`, or any `open_clip:<arch>:<pretrained>` / `hf-hub:<repo>` |
| text masking | `--text-mask` / `text_mask=True` | off | removes overlay text (EasyOCR + inpainting) before embedding; needs the `ocr` extra; ~0.3–1 s per image on CPU |
| device | `VSIM_DEVICE` | `cpu` | e.g. `cuda`, `mps` |
| cluster store | `VSIM_HOME` | `~/.vsim` | where `preprocess` writes `datasets/<id>/` |
| offline | `HF_HUB_OFFLINE=1` | unset | use cached weights only |

## How it works

More detail, and the reasoning behind each choice, is in [docs/](docs/).

**One embedding path.** Indexing, searching, clustering and assignment all go through
`vsim.embedding.embed`, which loads images (`vsim.images.load_image`), optionally masks
text, embeds them and L2-normalises the result. One shared path is the simplest guard
against train/serve skew.

**Search.** The index is a single `.npz` file. It holds the IDs, paths, embeddings,
zero-shot tag probabilities, calibration quantiles, an out-of-distribution threshold and
metadata, and it is loaded with `allow_pickle=False`. A query costs one matrix-vector
product. For 1,000 images × 512 dimensions that is microseconds, so an ANN index
(FAISS/HNSW) is not worth it below roughly 10⁵ images.

**Scores in [0, 1].** A raw cosine like 0.62 means nothing to a user. The score is the
empirical CDF of the dataset's own top-10 nearest-neighbour similarities. A score of 0.9
means "more similar than 90% of real neighbour pairs in this catalogue". The mapping is
monotonic, so the ranking is unchanged.

**Explanations.** CLIP rates every image against a small vocabulary of formats,
garments, colours and patterns (zero-shot). An explanation lists only the tags the query
and the result *share*, and adds "near-duplicate" or "different format" where that
applies.

**Out-of-distribution queries.** A query is flagged when its best cosine is below the
2nd percentile of the dataset's own nearest-neighbour cosines. Results are still
returned, but the explanation warns.

**Clustering.** K-means on the embeddings, with K chosen from 4–12. The rule: the best
silhouette among the K values whose clusterings agree across 5 seeds (ARI ≥ 0.8). Labels
are generated from the tags that distinguish each cluster from the dataset average. A
new image joins the cluster with the highest mean of its top-3 member similarities, or
becomes an `outlier` if it falls below the same out-of-distribution threshold.

## Evaluation

`experiments/eval_search.py` compares model configurations: CLIP, CLIP with text masking,
DINOv2, and DINOv2 with text masking. It uses checks that need no labels:

- **Near-duplicate recall:** pHash groups of the same photo. Are an image's group mates
  in its top results?
- **Augmentation recall@1:** query with a flipped, cropped or recompressed copy. Is the
  original ranked first?
- **Out-of-domain probes:** noise and solid colours. Are they flagged?
- **Timing:** indexing time and per-query latency.

It writes `experiments/results/search_eval.json`. `experiments/figures.py` renders the
calibration curve, the K-selection plot, the nearest-neighbour histogram and cluster
contact sheets into `lecture/assets/`.

```bash
uv sync --all-extras
HF_HUB_OFFLINE=1 uv run python experiments/eval_search.py data/images
uv run python experiments/figures.py data/images --dataset $(uv run vsim cluster data/images)
```

**The main finding so far.** CLIP reads rendered text very well. On catalogue images with
overlay text ("SALE -50%"), two *unrelated* banners with similar text become nearest
neighbours, while the *same photo* with different text does not. We call this the
overlay-text trap. Two fixes help, with different costs:

- Masking the text before embedding: ~25× slower to index.
- Switching to DINOv2, a self-supervised, image-only model: about the same speed.

See [docs/evaluation.md](docs/evaluation.md). A labelled precision@K evaluation on the
Fashion Product Images sample is planned.

## Project layout

```
src/vsim/
  images.py        folder listing + ID parsing, robust path/URL loading (size cap, timeouts, any mode -> RGB)
  models.py        CLIP / SigLIP2 / DINOv2 behind one Embedder interface, cached per process
  embedding.py     the single embed() pipeline used everywhere
  tags.py          zero-shot attribute tags -> explanations and cluster labels
  calibration.py   cosine -> [0, 1] score, out-of-distribution threshold
  textmask.py      optional OCR + inpainting text removal
  search.py        build_index(), search(), the .npz Index
  clustering.py    preprocess(), assign(), K selection, labelling, persistence
  cli.py           the `vsim` command
scripts/
  download_hm.py       H&M (Kaggle) sample -> data/images + data/labels.csv
  download_fashion.py  Fashion Product Images (HF) -> the same layout
experiments/
  eval_search.py      label-free comparison of model configurations
  figures.py          figures for README / slides
  plot_clusters.py    cluster mosaic + t-SNE map (gitignored output)
  results/            committed JSON results
tests/                pytest, with a fake embedder (no downloads, ~2 s)
docs/                 design notes (start at docs/README.md)
aueb_data_science_careers.html   the lecture deck (open in a browser)
```

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
```

See [AGENTS.md](AGENTS.md) for conventions and the invariants the code relies on.

## Limitations

- **Brute-force search** loads all embeddings in memory. That is fine up to about 10⁵
  images; beyond that, use FAISS/HNSW.
- **The tag vocabulary is fashion-oriented** (`tags.py`), so explanations on other
  domains will be vague. Edit `VOCAB`, then rebuild the index: the index stores the
  vocabulary and refuses a mismatch.
- **K-means assumes roughly round clusters.** On this kind of data the silhouette is low
  (about 0.15), so the clusters are useful groupings, not natural classes.
- **Scores are calibrated per index.** A 0.9 in one catalogue is not comparable to a 0.9
  in another.
- **Text masking is slow** on CPU and depends on EasyOCR detection quality.

## Documentation

| doc | what |
|---|---|
| [docs/README.md](docs/README.md) | index of the docs |
| [docs/architecture.md](docs/architecture.md) | modules, data flow, on-disk formats |
| [docs/data.md](docs/data.md) | datasets, the download script, the folder contract |
| [docs/search.md](docs/search.md) | retrieval, calibration, explanations, out-of-distribution |
| [docs/clustering.md](docs/clustering.md) | K selection, labels, assignment, persistence |
| [docs/evaluation.md](docs/evaluation.md) | how we evaluate without labels, and with them |
| [docs/decisions.md](docs/decisions.md) | design decisions, trade-offs, what we would do next |

## License

MIT.
