# Design decisions

Each entry gives the choice, the alternatives we considered, and why. Interviewers care
more about this page than about the code: it shows what you weighed, not just what you
built.

## Model: CLIP ViT-B/32 by default, DINOv2 S/14 one flag away

| | CLIP ViT-B/32 | DINOv2 S/14 |
|---|---|---|
| same product type in top 5 (H&M) | 0.55 | 0.62 |
| same colour in top 5 (H&M) | 0.22 | 0.15 |
| a bannered stranger beats the real match | 10% of queries | 0% |
| "same photo" recall (original catalogue) | 0.44 | 0.85 |
| zero-shot tags (explanations, cluster names) | yes | no |
| embed time, 998 images, CPU | 8 s | 13 s |

**Chosen:** CLIP as the default, because one model gives us embeddings *and* text-based
tags, which the brief asks for ("explain the result", "name the cluster").

**Trade-off we accept:** CLIP matches overlay text. On catalogues with heavy banners use
`--model dinov2-s14`. Tags are then still computed with CLIP, at the cost of a second
model in memory.

**Rejected:** fine-tuning. There are no labels to fine-tune on, it would not run on a
laptop CPU in review time, and the zero-shot baseline was not measured to be the
bottleneck until the evaluation said so. SigLIP2 is wired into the evaluation script as a
candidate. It is not a default until it has numbers.

## Text masking: optional, off by default

OCR + inpainting fixes most of CLIP's text problem (same-photo recall 0.44 → 0.85 on
the original catalogue; banner wins 10% → 4% on H&M), but makes embedding 13–27× slower,
pulls in EasyOCR and OpenCV, and stops two of the three OOD probes being flagged
([evaluation.md](evaluation.md)). DINOv2 does better for free. So masking is an
`ocr` extra and a flag, kept mainly to *show* the experiment.

## Search: brute-force cosine, no vector database

A matrix–vector product over a few thousand 512-d vectors takes microseconds (we measured
8–21 µs over 998 vectors). An ANN index (FAISS, HNSW) or a vector DB would add a dependency, a build step
and approximate results, for no measurable gain at this size.

**When we would change it:** above roughly 1 M vectors, or with a latency budget per
query that brute force cannot meet. Then FAISS IVF-PQ or HNSW, behind the same `rank`
function, evaluated on recall vs the exact search we have now.

## Score: empirical CDF, not raw cosine

Raw cosine depends on the model and the catalogue (CLIP median nn1 0.86, DINOv2 0.69).
A fixed threshold would be wrong for at least one of them. The CDF of the index's own
leave-one-out neighbour similarities gives a score in [0, 1] that reads as a percentile,
is monotonic (it never changes the ranking) and is recomputed for every index.

**Rejected:** a sigmoid with hand-picked temperature (needs tuning per model); softmax
over the results (makes scores depend on K and on the other results).

## Out-of-distribution: 2nd percentile of nn1, not the minimum

The minimum nn1 cosine is set by the single weirdest catalogue image, so it is too
permissive. A solid-colour square scored 0.657 against a minimum of 0.656. The 2nd
percentile (0.717) flags it while accepting 98% of catalogue images as in-distribution.
The threshold is stored with the index.

## Clustering: k-means + stability, not HDBSCAN

- **k-means** is simple, fast and easy to explain. Every image gets a cluster, and
  outliers are handled at *assignment* time with the same nn1-style threshold.
- **K by stability (ARI ≥ 0.8), then silhouette.** Silhouette alone is flat on this data
  (0.10–0.16) and picks noise. See [clustering.md](clustering.md).
- **Rejected:** HDBSCAN. It labels a large share of product images as noise and is
  sensitive to `min_cluster_size`, which we cannot tune without labels. Agglomerative
  clustering: fine, but no advantage we could measure, and harder to assign new points to.

## Cluster names: tag lift, not an LLM

Names come from which zero-shot tags are *over-represented* in a cluster versus the
whole dataset (lift), not which are most common. "Floral" in a cluster of 80% floral
dresses is informative. "Dress" in a catalogue of dresses is not. This is deterministic,
offline and free.

**Rejected:** captioning the cluster with an LLM or VLM. Better prose, but it is a
network call, non-deterministic, and harder to evaluate.

## Assignment: top-3 member similarity, not the centroid

A new image goes to the cluster whose three most similar members have the highest mean
similarity. Centroids of elongated clusters sit in empty space, and a single nearest
member is noisy. Top-3 is a middle ground. Below the threshold the image is an outlier.

## Storage: `.npz` + JSON, no pickle, atomic writes

- `.npz` with `allow_pickle=False` cannot execute code on load, so it is safe to load an
  index someone sent you. Pickle is not.
- Every file is written to a temp path and moved with `os.replace`. `clusters.npz` is
  written last, so a crash mid-`preprocess` leaves no half-valid dataset.
- Additions go to an append-only `additions.jsonl`, keyed by the sha256 of the image
  bytes. That makes `assign` idempotent: the same bytes give the same answer and no new
  line.
- The dataset id `ds_…` is a hash of the content and the settings. Re-running
  `preprocess` on the same folder is a no-op.

## One decode path, one embed path

Indexing and querying both call `images.load_image` and `embedding.embed`. A separate
"query preprocessing" function is the classic train/serve-skew bug: a different resize
or colour conversion silently degrades every query. One path makes that impossible.

## Known limitations

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

## What we would do next

In the order we would do them:

1. **Labelled evaluation on H&M:** precision@K by product type and by colour
   ([evaluation.md](evaluation.md#4-with-labels-precisionk)).
2. **A small human-judged set** (a few hundred query–result pairs) to check that the
   label-free metrics agree with people.
3. **Hybrid embedding:** DINOv2 for retrieval, CLIP for tags, picked by evaluation
   rather than by default.
4. **Serving:** a thin HTTP layer over `search` and `assign`, the index memory-mapped,
   and an ANN index only once size requires it.
5. **Monitoring:** the share of queries flagged out-of-distribution, and the score
   distribution over time. A drift in either is the first sign the catalogue or the
   users have changed.
