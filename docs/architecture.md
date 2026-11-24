# Architecture

```
                 ┌───────────────┐
 folder / URL ──▶│  images.py    │  list_images(), load_image(): any format → RGB, ≤1024 px
                 └──────┬────────┘
                        ▼
                 ┌───────────────┐   (optional) textmask.py: EasyOCR boxes → cv2 inpaint
                 │ embedding.py  │──▶ models.py: CLIP / SigLIP2 / DINOv2 → L2-normalised vectors
                 │   embed()     │──▶ tags.py:   zero-shot tag probabilities (CLIP text side)
                 └──────┬────────┘
              ┌─────────┴──────────┐
              ▼                    ▼
       ┌─────────────┐      ┌───────────────┐
       │ search.py   │      │ clustering.py │
       │ build_index │      │ preprocess    │
       │ search/rank │      │ assign        │
       └──────┬──────┘      └──────┬────────┘
              ▼                    ▼
        index.npz          $VSIM_HOME/datasets/<ds_id>/
                            clusters.npz, summary.json, additions.jsonl
```

`calibration.py` is used by both sides. It provides the [0, 1] score
(`reference_quantiles`, `calibrate`) and the out-of-distribution cut (`nn1_threshold`).

## Modules

**`images.py`.** All I/O for images.

- `list_images(folder)` returns `ImageFile(id, path)` sorted by numeric ID.
- `load_image(source)` accepts a path or an http(s) URL and does the following:
  - timeouts `(5, 15)` s
  - a 20 MB download cap
  - a `Content-Type` check, which rejects HTML pages pretending to be images
  - Pillow's decompression-bomb guard set to about 90 MP
  - EXIF rotation
  - alpha composited on white
  - `draft()` for cheap JPEG downscaling
  - a 1024 px cap

  Every failure raises `ImageLoadError`.

**`models.py`.** `Embedder` has `embed_images(images, batch_size)` and, for CLIP-like
models, `embed_texts`. The implementations are `OpenClipEmbedder` and `DinoEmbedder`.
`get_embedder(spec)` is `lru_cache`d, so a process loads each model once. Aliases are in
`MODEL_ALIASES`. When the main model has no text encoder (DINOv2), `get_tagger` falls
back to CLIP for tags.

**`embedding.py`.** `embed(images, model, text_mask)` returns
`Embedded(emb, tag_probs)`. Tags are always computed on the **unmasked** image, because
"sale banner" is itself useful information.

**`tags.py`.** `VOCAB` holds four categories (format, garment, colour, pattern). The
prompts are ensembled over two templates per category, and the softmax is taken per
category. `explain()` builds the one-line reason for a (query, result) pair.

**`search.py`.** The `Index` dataclass, with `save` and `load`. `load` is cached on
(path, mtime), so a rebuilt index is picked up automatically. `rank()` is separated from
`search()` so experiments can rank pre-computed embeddings.

**`clustering.py`.** `choose_k`, `make_labels`, `preprocess` and `assign`. See
[clustering.md](clustering.md).

**`cli.py`.** A thin argparse layer that imports lazily, so `vsim --help` is instant.

## On-disk formats

### `index.npz` (search)

| key | dtype / shape | meaning |
|---|---|---|
| `ids` | str (n,) | IDs as strings |
| `id_is_int` | bool (n,) | restores int IDs on load |
| `paths` | str (n,) | source file paths |
| `emb` | float32 (n, d) | L2-normalised embeddings |
| `tag_probs` | float32 (n, T) | zero-shot tag probabilities |
| `ref_q` | float32 (1001,) | quantiles of top-10 neighbour cosines (calibration) |
| `ood_threshold` | float32 () | 2nd percentile of nearest-neighbour cosine |
| `meta` | str () | JSON: model, text_mask, near_dup_cos, tag_names, version, timings, skipped files |

### `$VSIM_HOME/datasets/<ds_id>/` (clustering)

- **`clusters.npz`:** `ids`, `emb`, `tag_probs`, `assign` (int16), `labels`,
  `threshold` and `hashes` (sha256 of each image's bytes). It is written last and is
  the commit marker.
- **`summary.json`:** K, the K-selection table, the clusters (label, size, members),
  the model and timings.
- **`additions.jsonl`:** one line per assigned image, with hash, source, embedding,
  assignment and timestamp.

The dataset id is `ds_` followed by the first 16 hex characters of a sha256 over the
config (model, text_mask, K range, seeds) and every (ID, image-bytes hash) pair. With a
forced K, `_k<K>` is appended.

## Performance

These figures are for CPU, about 1k images, CLIP ViT-B/32:

| step | cost |
|---|---|
| model load | ~2–4 s, once per process |
| embed | ~10–15 ms per image (batch 32) |
| embed + text mask | ~0.3–1 s per image (OCR dominates) |
| search, excluding query embedding | ~5 µs (one 1k×512 mat-vec) |
| index size | ~2 KB per image |
