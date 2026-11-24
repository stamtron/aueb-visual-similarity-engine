# Search

```python
build_index(folder, index_path, model="clip-b32", text_mask=False, batch_size=32) -> Path
search(url, k, index_path) -> list[Result(id, score, explanation)]   # best first
```

## Pipeline

1. **Build.** Load every image in the folder, `embed` it, and compute three things over
   the dataset itself:
   - **`ref_q`:** 1001 quantiles of each image's top-10 leave-one-out neighbour cosines,
     used for calibration.
   - **`ood_threshold`:** the 2nd percentile of each image's nearest-neighbour cosine,
     used for out-of-distribution detection.
   - **`tag_probs`:** zero-shot tags, used for explanations.

   Then save everything to one `.npz`.
2. **Query.** Load the index (cached), then embed the query with **the model and options
   stored in the index**. You cannot accidentally query a CLIP index with DINOv2.
3. **Rank.** `cos = emb @ q`, then `argpartition` for the top K, then a stable sort by
   score.
4. **Score.** `calibrate(cos, ref_q)` gives a value in [0, 1].
5. **Explain.** `tags.explain(query_tags, result_tags, cos, score)` for each result.
6. **Flag.** If the best cosine is below `ood_threshold`, prefix every explanation with
   `low confidence: query unlike anything in the catalogue`.

## Why brute force

With n ≈ 1,000 images and d = 512 float32, the embedding matrix is about 2 MB, and a
query is one mat-vec, measured at ~5 µs. Embedding the query image (~15 ms) costs 3,000×
more than the search. ANN indexes (FAISS, HNSW) add a dependency, an approximation and a
build step, and pay off only at 10⁵–10⁶ items. The upgrade path is clear, because `rank`
is the only function that touches the matrix.

## The score: why not just return the cosine?

Raw CLIP cosines sit in a narrow, model-specific band:

- random noise vs a catalogue: ~0.56
- a typical (median) nearest neighbour: ~0.86
- near-duplicates: ≥ 0.95

"0.62" means nothing to a user, and the band moves when you change models.

So the score is the **empirical CDF of real neighbour similarities in this catalogue**:

```
score(c) = fraction of top-10 neighbour cosines in the dataset that are ≤ c
```

Its properties:

- **Monotonic.** The ranking is exactly the cosine ranking.
- **Bounded.** The CDF is anchored at 0 → 0 and 1 → 1.
- **Interpretable.** 0.9 means "closer than 90% of the neighbour pairs in this
  catalogue". It is calibrated per index and per model.
- **Cheap.** 1001 floats in the index, and `np.interp` at query time.

What it is **not** is a probability of relevance. That would need labels: with
`data/labels.csv` you could fit an isotonic regression from cosine to P(same pattern).

## Explanations

Each image gets a probability distribution over a small vocabulary (`tags.VOCAB`), with
a softmax per category:

| category | examples |
|---|---|
| format | studio product photo, model posing outdoors, runway show, sale banner, close-up |
| garment | dress, shirt, jeans, handbag, sneakers, … (24) |
| colour | black, white, red, navy, … (14) |
| pattern | plain, striped, floral, checked, animal print, … (10) |

A tag *applies* to an image if it is in the top 2 of its category and has probability
≥ 0.15. The explanation for a (query, result) pair is then built as follows:

- `identical image` if the cosine is ≥ 0.999
- `near-duplicate (cosine …)` if the cosine is ≥ the model's near-duplicate threshold
  (0.95 for CLIP B/32)
- `both: …` with the **intersection** of applying tags. Only the tags both images have
  are listed, so nothing is claimed that only one image has. When the score is below
  0.5, the wording becomes `only shares: …`.
- `different format (A vs B)` when the top formats differ, which is useful for
  "same dress, product shot vs worn"
- `overall visual similarity, no shared attribute tags` when nothing overlaps. This is
  an honest answer: the embedding agrees, but our vocabulary cannot say why.

Tags come from CLIP's text encoder. With DINOv2 as the main model, a CLIP model is
loaded alongside it just for tagging, so explanations look the same whichever model
ranks.

**Limitation:** the vocabulary is hand-written and fashion-specific. For another domain,
edit `VOCAB` and rebuild. The index stores `tag_names` and refuses to load with a
different vocabulary.

## Out-of-distribution queries

The threshold is the 2nd percentile of the dataset's own nearest-neighbour cosines, so
98% of catalogue images have a neighbour at least this close. A query whose *best* match
is below that does not look like the catalogue. We use a percentile rather than the
minimum because the minimum is too permissive: on the original data (CLIP B/32), the
minimum nearest-neighbour cosine was 0.656, and a plain solid-blue square scored 0.657
against the catalogue. The 2nd percentile (0.717) flags it.

We **still return K results**, because the caller asked for K. Every explanation is
prefixed with a low-confidence warning, and the caller decides what to show. Silent
filtering would hide exactly the cases a product team needs to see.

## Text masking (`text_mask=True`)

`textmask.remove_text` does the following:

1. Run EasyOCR on a ≤768 px copy of the image.
2. Scale the boxes back up and pad them by 6 px.
3. Inpaint them with `cv2.INPAINT_TELEA`.

The masked image is used for the **embedding only**. Tags still see the original. The
cost is ~0.3–1 s per image on CPU. [evaluation.md](evaluation.md) explains why the
option exists (the overlay-text trap) and how it compares with switching to DINOv2.

## Edge cases handled

| case | behaviour |
|---|---|
| `k` > n | returns n results |
| `k` ≤ 0, non-int or bool | `ValueError` |
| query URL is HTML, too large, times out, or returns 404 | `ImageLoadError` with the reason |
| RGBA, palette, CMYK or EXIF-rotated query | normalised to upright RGB, same as at index time |
| index file missing | `FileNotFoundError` telling you to build it |
| index built with another tag vocabulary | `ValueError`: rebuild |
| index rebuilt while the process runs | cache keyed on mtime, so it reloads |
