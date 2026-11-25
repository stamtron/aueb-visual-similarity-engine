# Usage

The full reference for the command line, the Python API and the settings. To install the
engine and get data, see the [README](../README.md#quick-start) and [data.md](data.md). If
you are new to the terminal or on Windows, start with [getting-started.md](getting-started.md).

## Install

Requirements:

- Python 3.10 or newer (3.12 is pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- About 2 GB of disk for PyTorch and the model weights

```bash
uv sync                      # core: search + clustering
uv sync --all-extras         # + OCR text masking, evaluation and figures, data download
```

The extras:

| extra | adds | needed for |
|---|---|---|
| `ocr` | EasyOCR, OpenCV | `--text-mask` |
| `eval` | imagehash, matplotlib | `experiments/` |
| `data` | pyarrow, kaggle | `scripts/download_*.py` |

Without uv:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[ocr,eval,data]"
```

The first real run downloads the model weights:

- CLIP ViT-B/32: about 350 MB, via open_clip
- DINOv2 S/14: about 90 MB, via torch.hub

Both are cached afterwards, and after that the engine works offline (`HF_HUB_OFFLINE=1`).

## Command line

Every command takes `-v` for progress logs. Wherever a command takes an image, it can
be a local path or an `https://` URL.

```bash
# --- search ---
vsim index data/images --out index.npz [--model clip-b32|dinov2-s14|...] [--text-mask]
vsim search <path-or-url> -k 5 --index index.npz

# --- clustering ---
vsim cluster data/images [--k 6] [--model ...] [--text-mask]   # prints a dataset id, e.g. ds_3f2a...
vsim clusters ds_3f2a...                                         # labels, sizes, example members, K table
vsim assign <path-or-url> ds_3f2a... [--dry-run]                 # prints a label, or "outlier"
```

What to expect:

- `search` reads the model and options stored in the index, so you never pass `--model`
  twice.
- `cluster` is cached by content: running it again on the same folder returns the same
  id instantly. Changing one image changes the id.
- `assign` saves each new image (unless you pass `--dry-run`), so later assignments can
  see earlier ones. Assigning the same image twice is a no-op.

## Python API

```python
from vsim import build_index, search, preprocess, assign

# search
build_index("data/images", "index.npz", model="clip-b32", text_mask=False)
for r in search("query.jpg", k=5, index_path="index.npz"):  # a path or an https:// URL
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

## Experiments and figures

```bash
uv sync --all-extras
HF_HUB_OFFLINE=1 uv run python experiments/eval_search.py data/images
uv run python experiments/figures.py data/images --dataset $(uv run vsim cluster data/images)
uv run python experiments/plot_clusters.py $(uv run vsim cluster data/images) data/images
```

- `eval_search.py` compares model configurations with label-free checks and writes
  `experiments/results/search_eval.json` ([evaluation.md](evaluation.md)).
- `figures.py` renders the calibration curve, the K-selection plot, the
  nearest-neighbour histogram and contact sheets.
- `plot_clusters.py` renders a mosaic of each cluster and a t-SNE map of the catalogue.

Figures go to `lecture/assets/`, which is gitignored because they contain dataset photos.

The `$(...)` syntax works in bash, zsh and PowerShell.

## Development

```bash
uv run pytest -q             # no network, no model download (a fake embedder), ~2 s
uv run ruff check .
```

See [AGENTS.md](../AGENTS.md) for conventions and the invariants the code relies on.
