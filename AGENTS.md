# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, …) and for human contributors.
Read this before changing code. [README.md](README.md) covers usage, and
[docs/](docs/) covers the reasoning behind the design.

## What this repo is

`vsim` is a small visual similarity engine with two features:

- **Search:** `build_index` and `search`.
- **Clustering:** `preprocess` and `assign`.

It is used as a worked take-home example in a university careers lecture
(`slides/aueb_data_science_careers.html`) and walked through in
`slides/vsim_interview_example.html`. Clarity and honest evaluation matter more than
features. Keep it small, readable and runnable on a laptop CPU.

## Setup and commands

```bash
uv sync --all-extras          # Python 3.12 via .python-version
uv run pytest -q              # must pass; ~2 s, no network, no model downloads
uv run ruff check .           # line length 100
uv run vsim --help            # CLI entry point: src/vsim/cli.py:main
```

Run experiments only when the task needs them. They download model weights and embed
real images, which takes minutes:

```bash
HF_HUB_OFFLINE=1 uv run python experiments/eval_search.py data/images
uv run python experiments/figures.py data/images --dataset <ds_id>
```

## Layout

| path | role |
|---|---|
| `src/vsim/images.py` | the **only** place images are read from disk or URL |
| `src/vsim/embedding.py` | the **only** embedding pipeline (`embed`) |
| `src/vsim/models.py` | model aliases, `Embedder` interface, per-process model cache |
| `src/vsim/tags.py` | zero-shot tag vocabulary → explanations and cluster labels |
| `src/vsim/calibration.py` | cosine → [0, 1] score; out-of-distribution threshold |
| `src/vsim/search.py` | `Index` (.npz), `build_index`, `search`, `rank` |
| `src/vsim/clustering.py` | `choose_k`, `make_labels`, `preprocess`, `assign` |
| `src/vsim/textmask.py` | optional EasyOCR + inpainting (lazy import, `ocr` extra) |
| `scripts/` | data download, kept out of the package |
| `experiments/` | evaluation and figures; `results/*.json` is committed |
| `tests/` | pytest; `conftest.py` swaps in a fake embedder |

## Invariants: do not break these

1. **One decode path, one embed path.** Indexing and querying must both go through
   `images.load_image` and `embedding.embed`. Never add a second preprocessing route for
   queries: that is the train/serve-skew bug this design exists to prevent.
2. **No pickle.** Indexes and cluster stores are `.npz`, loaded with
   `allow_pickle=False`, plus JSON or JSONL. Do not store Python objects.
3. **Atomic writes.** Write to a temp file, then `os.replace`. In `preprocess`,
   `clusters.npz` is written **last** and acts as the commit marker.
4. **IDs come from file names** (`123.jpg` → `123`). They are never list positions.
   Duplicate IDs are an error.
5. **Scores stay in [0, 1] and are monotonic in cosine.** Calibration must never change
   the ranking.
6. **Index metadata travels with the index.** `search` reads the model and `text_mask`
   from `index.meta`. If you change the tag vocabulary or the embedding semantics, make
   old indexes fail loudly (see the `tag_names` check in `_load_cached`).
7. **`assign` is idempotent.** The same image bytes must give the same answer with no
   new write. Additions are appended to `additions.jsonl`.
8. **Heavy deps are lazy.** `easyocr`, `cv2`, `imagehash` and `matplotlib` are imported
   only where they are used, so the core install stays small.

## Testing conventions

- Tests must not touch the network or download weights. Use the autouse `fake_models`
  fixture: it embeds an image as its 8×8 thumbnail and sets `VSIM_HOME` to a temp dir.
- Use the `dataset` fixture (30 synthetic images in 3 colour families) for anything
  that needs a folder.
- New behaviour needs a test. Bug fixes need a test that fails before the fix.
- Model-quality claims belong in `experiments/`, with numbers, not in unit tests.

## Style

- Python ≥ 3.10 with `from __future__ import annotations`, type hints on public
  functions, dataclasses or NamedTuples for results.
- Module docstrings explain **why**, not just what. Keep that habit: this code is read
  by students.
- Prefer small and explicit over clever. No new runtime dependency without a clear
  reason, written in the PR or commit.
- Ruff, line length 100.

## Data, secrets and what must never be committed

These paths are gitignored. Keep it that way:

- `data/` (raw parquet files, images, `labels.csv`)
- `experiments/cache/`, `*.npz`
- `.env`, `kaggle.json`
- `assignments/`, `Data Science Challenge*.pdf`, `lecture/assets/`: private
  take-home material and figures derived from a non-public dataset

The repo is **public**. Before committing, check `git status` for images, PDFs, tokens or
anything from the folders above. Never paste the content of private briefs into code,
docs or commit messages.

## When you change things

- Changed a CLI flag or API? Update `docs/usage.md` (and the README quick start if it shows it) and the relevant
  `docs/*.md`.
- Changed defaults (model, K range, thresholds)? Re-run `experiments/eval_search.py`,
  update `experiments/results/`, and update the numbers in `docs/evaluation.md`.
- Don't push, open PRs or publish anything without explicit approval from the
  maintainer.
