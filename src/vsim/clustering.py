"""Assignment 2: clustering with labels, and online assignment of new images.

Function 1  `preprocess(folder, ...)`   -> dataset_id (str)
Function 2  `assign(url, dataset_id)`   -> Assignment (its `.label` is a cluster label or "outlier")

Design choices, each tested in `experiments/`:

* K is chosen, not hard-coded: k-means for K in 4..12 with 5 seeds each. We keep the K
  with the best silhouette among those whose seed-to-seed agreement (mean pairwise ARI)
  is >= 0.8. Silhouette alone is flat and low on this data, so stability decides.
* Labels are automatic: the cluster's dominant image format plus its most *distinctive*
  garments (cluster mean tag probability minus dataset mean), made unique.
* Outliers: a new image is an outlier if its best cosine to any member is below the 2nd
  percentile of the dataset's own nearest-neighbour cosines. Otherwise it joins the
  cluster with the highest mean of its top-3 member similarities (robust to one stray
  member, unlike centroid distance for these elongated clusters).
* The dataset id is a content hash of the image bytes plus the config, so running
  `preprocess` twice on the same folder is a cache hit, and editing one image changes it.
* Assigned images are persisted (`additions.jsonl`), keyed by the image's hash, so
  assigning the same image twice is idempotent, and later assignments see earlier ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import __version__
from . import tags as tagmod
from .calibration import nn1_threshold
from .embedding import embed
from .images import ImageLoadError, list_images, load_image, load_image_bytes
from .models import DEFAULT_MODEL, resolve

log = logging.getLogger(__name__)

OUTLIER = "outlier"
K_RANGE = range(4, 13)
N_SEEDS = 5
MIN_ARI = 0.8
TOP_M = 3
SECOND_TAG_SHARE = 0.5  # name a 2nd product type if its mean prob is >= half the first's


def store_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("VSIM_HOME", Path.home() / ".vsim")) / "datasets"


@dataclass
class Assignment:
    label: str  # cluster label, or "outlier"
    cluster: int  # -1 for outliers
    similarity: float  # best cosine to any member
    threshold: float  # outlier cut-off used
    nearest_ids: list
    already_known: bool = False  # True if this exact image was seen before (no new write)

    @property
    def is_outlier(self) -> bool:
        return self.cluster == -1

    def __str__(self) -> str:
        return self.label


# ----------------------------------------------------------------------------- choosing K
def choose_k(emb: np.ndarray, k_range=K_RANGE, n_seeds: int = N_SEEDS, min_ari: float = MIN_ARI):
    """Returns (best_k, table) where table rows are {k, silhouette, ari}."""
    k_range = [k for k in k_range if 2 <= k < len(emb)]
    table = []
    for k in k_range:
        runs = [KMeans(k, n_init=3, random_state=s).fit_predict(emb) for s in range(n_seeds)]
        sil = float(np.mean([silhouette_score(emb, r, metric="cosine") for r in runs]))
        aris = [adjusted_rand_score(runs[i], runs[j])
                for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        table.append({"k": k, "silhouette": round(sil, 4), "ari": round(float(np.mean(aris)), 4)})
    stable = [r for r in table if r["ari"] >= min_ari] or [max(table, key=lambda r: r["ari"])]
    return max(stable, key=lambda r: r["silhouette"])["k"], table


# ----------------------------------------------------------------------------- labels
def make_labels(tag_probs: np.ndarray, assign: np.ndarray) -> list[str]:
    names = tagmod.TAG_NAMES
    cats = np.array([n.split(":", 1)[0] for n in names])
    terms = [n.split(":", 1)[1] for n in names]
    global_mean = tag_probs.mean(0)
    k = int(assign.max()) + 1
    labels, lifts = [], []
    for c in range(k):
        m = tag_probs[assign == c].mean(0)
        lift = m - global_mean
        fmt = terms[max(np.flatnonzero(cats == "format"), key=lambda i: m[i])]
        first, second = sorted(np.flatnonzero(cats == "garment"), key=lambda i: -lift[i])[:2]
        # A second name only if it describes a real share of the cluster, so a pure
        # cluster of watches is "watch", not "watch & hoodie".
        garments = [first] + ([second] if m[second] >= SECOND_TAG_SHARE * m[first] else [])
        labels.append(f"{fmt}: " + " & ".join(terms[i] for i in garments))
        lifts.append(lift)
    # Disambiguate duplicates with the most distinctive colour/pattern, then a counter.
    for c, lab in enumerate(labels):
        dupes = [d for d, other in enumerate(labels) if other == lab]
        if len(dupes) > 1:
            for d in dupes:
                extra = np.flatnonzero(np.isin(cats, ["colour", "pattern"]))
                best = max(extra, key=lambda i: lifts[d][i])
                labels[d] = f"{lab} ({terms[best]})"
    seen: dict[str, int] = {}
    for c, lab in enumerate(labels):
        seen[lab] = seen.get(lab, 0) + 1
        if seen[lab] > 1:
            labels[c] = f"{lab} #{seen[lab]}"
    return labels


# ----------------------------------------------------------------------------- function 1
def _dataset_id(files, model: str, text_mask: bool) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"model": resolve(model), "text_mask": text_mask, "v": 1,
                         "tags": tagmod.TAG_NAMES,
                         "k_range": list(K_RANGE), "seeds": N_SEEDS}).encode())
    for f in files:
        h.update(str(f.id).encode() + b"\0" + hashlib.sha256(f.path.read_bytes()).digest())
    return "ds_" + h.hexdigest()[:16]


def preprocess(folder: str | Path, model: str = DEFAULT_MODEL, text_mask: bool = False,
               root: str | Path | None = None, k: int | None = None) -> str:
    """Function 1: cluster the folder, label the clusters, persist, return a dataset id."""
    files = list_images(folder)
    ds_id = _dataset_id(files, model, text_mask) + (f"_k{k}" if k else "")
    out = store_root(root) / ds_id
    if (out / "clusters.npz").is_file():
        log.info("dataset %s already processed (cache hit)", ds_id)
        return ds_id

    t0 = time.perf_counter()
    images, ok = [], []
    for f in files:
        try:
            images.append(load_image(f.path))
            ok.append(f)
        except ImageLoadError as e:
            log.warning("skipping %s: %s", f.path, e)
    e = embed(images, model, text_mask)
    if k is None:
        k, table = choose_k(e.emb)
    else:
        table = []
    assign = KMeans(k, n_init=10, random_state=0).fit_predict(e.emb)
    labels = make_labels(e.tag_probs, assign)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "clusters.tmp.npz",
        ids=np.array([str(f.id) for f in ok]),
        emb=e.emb,
        tag_probs=e.tag_probs,
        assign=assign.astype(np.int16),
        labels=np.array(labels),
        threshold=np.float32(nn1_threshold(e.emb, 2.0)),
        hashes=np.array([hashlib.sha256(f.path.read_bytes()).hexdigest() for f in ok]),
    )
    summary = {
        "dataset_id": ds_id,
        "folder": str(Path(folder).resolve()),
        "model": resolve(model),
        "text_mask": text_mask,
        "k": k,
        "k_selection": table,
        "clusters": [
            {"cluster": c, "label": labels[c],
             "size": int((assign == c).sum()),
             "members": [ok[i].id for i in np.flatnonzero(assign == c)]}
            for c in range(k)
        ],
        "package_version": __version__,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    os.replace(out / "clusters.tmp.npz", out / "clusters.npz")  # written last = commit marker
    log.info("dataset %s: %d images, K=%d in %.1fs", ds_id, len(ok), k, summary["seconds"])
    return ds_id


# ----------------------------------------------------------------------------- function 2
def load_dataset(dataset_id: str, root: str | Path | None = None) -> dict:
    d = store_root(root) / dataset_id
    if not (d / "clusters.npz").is_file():
        raise KeyError(f"Unknown dataset id {dataset_id!r} (looked in {d})")
    with np.load(d / "clusters.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["summary"] = json.loads((d / "summary.json").read_text())
    data["dir"] = d
    adds = d / "additions.jsonl"
    data["additions"] = [json.loads(line) for line in adds.read_text().splitlines()] \
        if adds.is_file() else []
    return data


def assign(url: str, dataset_id: str, root: str | Path | None = None,
           persist: bool = True) -> Assignment:
    """Function 2: put the image at `url` into an existing cluster, or call it an outlier."""
    ds = load_dataset(dataset_id, root)
    labels = [str(x) for x in ds["labels"]]
    img, raw = load_image_bytes(url)
    digest = hashlib.sha256(raw).hexdigest()

    # Idempotency: an image we already hold (original member or earlier addition).
    hit = np.flatnonzero(ds["hashes"] == digest)
    if hit.size:
        c = int(ds["assign"][hit[0]])
        return Assignment(labels[c], c, 1.0, float(ds["threshold"]),
                          [_id(ds["ids"][hit[0]])], already_known=True)
    for a in ds["additions"]:
        if a["hash"] == digest:
            return Assignment(**{**a["assignment"], "already_known": True})

    s = ds["summary"]
    q = embed([img], s["model"], s["text_mask"]).emb[0]
    emb, member_cluster = ds["emb"], ds["assign"].astype(int)
    ids = [_id(x) for x in ds["ids"]]
    added = [a for a in ds["additions"] if a["assignment"]["cluster"] >= 0]
    if added:  # earlier assignments become members for later ones
        emb = np.vstack([emb, np.array([a["emb"] for a in added], np.float32)])
        member_cluster = np.concatenate([member_cluster, [a["assignment"]["cluster"] for a in added]])
        ids += [a["source"] for a in added]

    cos = emb @ q
    best = float(cos.max())
    threshold = float(ds["threshold"])
    nearest = [ids[i] for i in np.argsort(-cos)[:5]]
    if best < threshold:
        result = Assignment(OUTLIER, -1, round(best, 4), round(threshold, 4), nearest)
    else:
        scores = []
        for c in range(len(labels)):
            cc = np.sort(cos[member_cluster == c])[::-1][:TOP_M]
            scores.append(cc.mean() if cc.size else -np.inf)
        c = int(np.argmax(scores))
        result = Assignment(labels[c], c, round(best, 4), round(threshold, 4), nearest)

    if persist:
        rec = {"hash": digest, "source": str(url), "emb": [round(float(x), 6) for x in q],
               "assignment": asdict(result), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(ds["dir"] / "additions.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return result


def _id(x) -> int | str:
    s = str(x)
    return int(s) if s.isdigit() else s
