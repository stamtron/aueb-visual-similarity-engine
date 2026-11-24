# vsim: a small visual similarity engine

**Point it at a folder of images. Find look-alikes, and get named clusters, on a laptop CPU.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)
![CPU only](https://img.shields.io/badge/GPU-not%20needed-555)
![no vector DB](https://img.shields.io/badge/vector%20DB-not%20needed-555)
![License: MIT](https://img.shields.io/badge/license-MIT-555)

This is the worked take-home example for the AUEB MSc Data Science careers lecture
([slides, PDF](slides/aueb_data_science_careers.pdf)). It is small enough to read in an
evening, and every design claim is backed by an experiment you can re-run.

## What it does

**🔎 Search.** Query with any image (a path or a URL) and get the most similar images,
each with a **score in [0, 1]** and an **explanation** of what they share:

```console
$ vsim search dress.jpg -k 3 --index index.npz     # a dress that is not in the index
707363002  0.608  both: studio product photo, floral; different format (studio product photo vs sale banner)
569523002  0.532  both: studio product photo, close-up of a product, floral
666352003  0.523  both: studio product photo, close-up of a product, dress, blue
```

**🗂️ Cluster.** Group the folder into **automatically named** clusters (it picks K
itself), then file new images into a cluster, or flag them as **outliers**:

```console
$ vsim cluster data/images                          # 923 H&M photos
ds_1abc6c700def80b5
$ vsim clusters ds_1abc6c700def80b5                 # K=9, chosen automatically (trimmed)
[0] studio product photo: jewellery  (n=54)
[2] studio product photo: dress & tunic  (n=146)
[4] studio product photo: sneakers  (n=46)
[7] close-up of a product: shorts & trousers  (n=132)
...
$ vsim assign sneaker.jpg ds_1abc6c700def80b5      # a sneaker that was not clustered
studio product photo: sneakers
```

Real output on 923 H&M catalogue photos (IDs are H&M article ids), on a laptop CPU:
indexing took about 17 s.

## How it works

```mermaid
flowchart LR
    A[folder or URL] --> B[load_image]
    B --> C["embed()<br/>CLIP or DINOv2"]
    C --> D[(index.npz)]
    C --> E[(clusters)]
    D --> F[search: top K + score + explanation]
    E --> G[assign: cluster or outlier]
```

- **One path for everything.** Indexing, querying, clustering and assignment all decode
  and embed images the same way, so queries can never drift from the index
  (train/serve skew).
- **Brute-force cosine.** One matrix-vector product per query. Below about 10⁵ images
  a vector database is not worth it.
- **Scores that mean something.** 0.9 means "more similar than 90% of real neighbour
  pairs in this catalogue". The ranking is unchanged.
- **K chosen by stability.** K-means for K = 4–12, keeping only the K values that
  give the same clusters whatever the random seed; cluster names come from
  zero-shot CLIP tags.

The reasoning behind each choice is in [docs/](docs/README.md).

## Quick start

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/stamtron/aueb-visual-similarity-engine.git
cd aueb-visual-similarity-engine
uv sync
uv run pytest -q                                    # ~2 s, no downloads

uv run vsim index path/to/images --out index.npz    # first run downloads CLIP (~350 MB)
uv run vsim search path/to/query.jpg -k 5 --index index.npz

uv run vsim cluster path/to/images                  # prints a dataset id: ds_...
uv run vsim assign path/to/new.jpg ds_...
```

Any folder of images works; file names become IDs (`123.jpg` → `123`). For the reference
dataset, a 1000-image sample of the H&M catalogue from Kaggle, follow
[docs/data.md](docs/data.md). It takes three commands and works on Windows too.

The full CLI, the Python API and the settings are in [docs/usage.md](docs/usage.md).

## The interesting part

**CLIP reads text.** On catalogue images with overlay text ("SALE -50%"), two
*unrelated* banners with similar text become nearest neighbours, while the *same photo*
with a different banner does not. We found this with label-free checks (near-duplicate
recall, augmentation recall, out-of-domain probes), then measured two fixes (CPU):

| configuration | near-duplicate recall | time to embed the folder |
|---|---|---|
| CLIP (baseline) | 0.44 | 6 s |
| CLIP + text masking (OCR + inpainting) | 0.85 | 162 s (~27× slower) |
| DINOv2 (image-only, self-supervised) | 0.85 | 8 s |

CLIP stays the default because its zero-shot tags power the explanations and the cluster
names; DINOv2 is one flag away (`--model dinov2-s14`). These numbers come from the original
take-home images, and they will be re-measured on H&M.

How we evaluate without labels, and what we found: [docs/evaluation.md](docs/evaluation.md).

## Documentation

| doc | read it for |
|---|---|
| [usage.md](docs/usage.md) | install options, every command, the Python API, settings |
| [data.md](docs/data.md) | getting the H&M sample (or a no-account alternative) |
| [architecture.md](docs/architecture.md) | modules, on-disk formats, repo layout |
| [search.md](docs/search.md) | retrieval, the [0, 1] score, explanations, outliers |
| [clustering.md](docs/clustering.md) | choosing K, naming clusters, assignment |
| [evaluation.md](docs/evaluation.md) | evaluating without labels, and with them |
| [decisions.md](docs/decisions.md) | trade-offs, known limitations, what we would do next |
| [AGENTS.md](AGENTS.md) | contributing: conventions and the invariants the code relies on |

## License

[MIT](LICENSE). The H&M images are not included and are subject to the
[competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
