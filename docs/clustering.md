# Clustering

```python
preprocess(folder, model="clip-b32", text_mask=False, root=None, k=None) -> "ds_…"
assign(url, dataset_id, root=None, persist=True) -> Assignment
```

`Assignment` has these fields: `label`, `cluster` (−1 for outliers), `similarity` (best
cosine), `threshold`, `nearest_ids`, `already_known` and `is_outlier`.

## 1. Choosing K

**The algorithm:** k-means on the L2-normalised embeddings, with K tried over 4–12 and
5 seeds per K.

For each K we record:

- **silhouette**, with cosine distance, averaged over the seeds
- **stability**, the mean pairwise adjusted Rand index (ARI) between the 5 seeded runs

**The rule:** among the K values with ARI ≥ 0.8, pick the best silhouette. If none
qualifies, pick the most stable K.

**Why not silhouette alone?** On product-image embeddings the silhouette is low and flat
(about 0.10–0.16 across K). The data has no crisp natural clusters, so "maximise
silhouette" picks noise. Stability asks a more useful question: would a colleague who
re-ran this get the same clusters? Small K values are usually both stable and the most
separated. The K-selection table is stored in `summary.json` and printed by
`vsim clusters`, so the choice is auditable.

**Why the range starts at 4:** two or three clusters are too coarse to be useful for
browsing. You can force any K with `--k` (or `k=`). That changes the dataset id
(`_k<K>` suffix).

## 2. Naming clusters

The labels need no human input. They come from the zero-shot tags in
[search.md](search.md#explanations):

1. **Format.** Take the format with the highest mean probability in the cluster, e.g.
   `runway show`.
2. **Garments.** Take the two garments with the biggest **lift**, meaning cluster mean
   minus dataset mean. Lift matters: "dress" might be the most likely garment
   *everywhere*, so the raw maximum would name every cluster "dress". Lift names what
   is *distinctive*.
3. **Uniqueness.** If two clusters end up with the same label, add each one's most
   distinctive colour or pattern, e.g. `(floral)`. If they still collide, add `#2`.

Example: `runway show: dress & handbag`. The labels are a starting point for a human
curator, not ground truth.

## 3. Assigning a new image

1. **Already known?** The sha256 of the image bytes is compared against the original
   members and earlier additions. On a hit, the stored answer comes back with
   `already_known=True` and nothing is written. That makes it idempotent.
2. **Embed** with the dataset's stored model and options.
3. **Outlier test.** If the best cosine to *any* member is below the dataset's
   out-of-distribution threshold (2nd percentile of nearest-neighbour cosine), the image
   is an `outlier` (cluster −1).
4. **Otherwise**, pick the cluster with the highest **mean of its top-3 member
   similarities**.
   - We don't use the nearest centroid because these clusters are elongated and mixed.
     A centroid can sit far from every real member, so the nearest centroid often
     disagrees with what the image visibly resembles.
   - We don't use the single nearest member (1-NN) because it is fragile: one stray
     member can capture the image. Top-3 is a cheap middle ground.
5. **Persist** (unless `persist=False` / `--dry-run`). One JSON line is appended to
   `additions.jsonl`. Later assignments include earlier non-outlier additions as
   members, so the clusters grow online without re-running k-means.

## Persistence and ids

- The **dataset id** is a content hash of the config and every image's bytes. Running
  `preprocess` twice on the same folder is a cache hit. Changing one pixel of one image
  gives a new id, so a stale clustering is never silently reused.
- **Writes are crash-safe.** `clusters.npz` goes to a temp name and is renamed last. A
  directory without it is treated as not processed.
- **Store location:** `$VSIM_HOME/datasets/` (default `~/.vsim/datasets/`).

## What we would change for production

- Re-cluster periodically, and when additions exceed about 10–20% of the members.
  Online assignment never splits or merges clusters.
- Replace k-means with HDBSCAN on a UMAP projection if the data has real density
  structure. That gives a native "noise" label, at the cost of more knobs.
- Let curators rename clusters and store the overrides by cluster id.
