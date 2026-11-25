# Evaluation

"It looks right on five queries" is not an evaluation. This page explains how we measure
search quality, first **without relevance labels** and then with them, what we found,
and where the method itself breaks.

Everything here comes from one script you can re-run:

```bash
HF_HUB_OFFLINE=1 uv run python experiments/eval_search.py data/images
# -> experiments/results/search_eval.json
```

The numbers are for the H&M sample ([data.md](data.md)): 998 studio photos, 20 product
types × 50 (one type has 48), on a laptop CPU. The script reads `data/labels.csv` for
precision@K. The engine itself never sees the labels.

## 1. Checks that need no labels

### Augmentation recall@1

Take an indexed image, transform it, and query with the copy. The original should be
rank 1. Transforms: greyscale, downscale to 128 px, JPEG quality 10, horizontal mirror,
70% centre crop. This tests robustness to things a user's upload will actually do. We
use 50 random images per transform.

### The banner trap: does the model read the text?

CLIP was trained on image–caption pairs, so it can read. If a catalogue is full of
"SALE" banners, does it match the banner instead of the product?

H&M photos carry no overlay text, so we add it. The same red "SALE -50%" strip is
stamped on 50 products, which go into the gallery as distractors. Then we query with 50
*other* products wearing that same banner. The right answer is the query's own clean
original. We report:

- **banner R@1:** how often the clean original is still rank 1
- **banner wins:** how often at least one bannered stranger outranks the clean original,
  that is, how often the text beat the product

Building the trap synthetically gives known answers, and it works on any catalogue.

### Near-duplicate recall (and why it does not apply here)

Catalogues often reuse one photo with a different crop, banner or language. A perceptual
hash (pHash, Hamming distance ≤ 14) finds these groups, and the same photo must count as
similar whatever "similar" means. For an image in a group of size *g*, we ask: are its
*g − 1* mates in its top *g − 1* results?

**On H&M this check finds nothing to measure.** The sample keeps one colour variant per
product, so there are no reused photos: the closest pair of images is still 4 bits
apart. At threshold 14, pHash instead chains look-alike white-background shots into 20
groups covering 805 of the 998 images. Treating those as "duplicates" would report a
meaningless 80%. The script now detects this (any group larger than 10 images) and
reports `n/a`.

The lesson generalises: a weak labeller has to be checked against the data before you
trust the metric it produces.

### Sanity checks

- **Out-of-distribution:** random noise, a solid blue square, black text on white. Each
  should be flagged by the nn1 threshold ([search.md](search.md)).
- **Timing:** embedding time for the whole folder and per-query search latency.

## 2. With labels: precision@K

For each image, what share of its top K results has the same label? We average per class
first, then over classes (macro average), so a large class like "Black" does not
dominate. We use two labels:

- **product type** (`product_type_name`, 20 classes): does the model group by *what* it is?
- **colour** (`perceived_colour_master_name`, 18 classes): or by *what colour* it is?

At random, precision would be about 0.05 for both.

## 3. Results (H&M, 998 images, CPU)

| config | type P@5 | type P@10 | colour P@5 | colour P@10 | aug R@1 (mean) | banner R@1 | banner wins | OOD flagged | embed 998 imgs |
|---|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 0.55 | 0.52 | **0.22** | **0.19** | 0.89 | 0.90 | 0.10 | 3/3 | 8 s |
| CLIP + text masking | 0.55 | 0.51 | 0.21 | 0.19 | 0.88 | 0.92 | 0.04 | 1/3 | 110 s |
| DINOv2 S/14 | **0.62** | **0.59** | 0.15 | 0.14 | **0.98** | **1.00** | **0.00** | 3/3 | 13 s |
| DINOv2 + text masking | 0.62 | 0.58 | 0.14 | 0.14 | 0.96 | 0.98 | 0.00 | 1/3 | 116 s |

Augmentation recall@1 per transform:

| config | greyscale | downscale | JPEG q10 | mirror | crop 70% |
|---|---|---|---|---|---|
| CLIP | 0.94 | 0.94 | 0.76 | 1.00 | 0.82 |
| DINOv2 | 1.00 | 1.00 | 0.92 | 1.00 | 0.98 |

Search latency was 8–21 µs per query in every configuration (brute-force cosine over
998 vectors). SigLIP2 is in the script but needs the `transformers` package, so it was
skipped in this run.

## 4. What we learned

**CLIP still reads text, and DINOv2 does not.** In 10% of the bannered queries, a
stranger wearing the same banner outranked the query's own clean photo under CLIP.
DINOv2 never let that happen (0 of 50). Text masking cuts CLIP's rate to 4%, but not to
zero, because the inpainted patch still differs from the original photo.

**Text masking costs far more than it gives.**

- Embedding is about **13× slower** (8 s → 110 s), and it needs a heavy optional
  dependency.
- On H&M there is no real overlay text, so precision does not move at all.
- It breaks the out-of-distribution check. Once the text is inpainted, "black text on
  white" becomes a near-blank white image, which looks like a studio shot (cosine 0.93,
  above every real nearest neighbour's 2nd percentile). The solid blue square is also
  no longer flagged.
- It can erase prints and slogans on garments, which are part of the product.

Masking is a fix for catalogues that really have banners. Even there, switching to
DINOv2 removes the problem more cheaply.

**DINOv2 groups by *what*, CLIP groups by *colour*.** DINOv2 is clearly better on
product type (P@5 0.62 vs 0.55) and on robustness (JPEG 0.92 vs 0.76, crop 0.98 vs
0.82). CLIP is better on colour (0.22 vs 0.15). Neither answer is "right": which one a
shopper wants is a product decision, and the labels let us see the trade-off instead of
guessing.

**Why CLIP is still the default.** CLIP gives us zero-shot **tags** (pattern, colour,
garment), which drive the explanations and the cluster names, and it does not need a
second model. DINOv2 is one flag away: `vsim index … --model dinov2-s14`. The trade-off
is written down in [decisions.md](decisions.md).

**Raw cosine is not a score.** CLIP nn1 cosines sit in a narrow band (2nd percentile
0.79, median 0.93), and random noise still reaches 0.67. DINOv2's band is very
different (p2 0.53, median 0.84). This is why we calibrate per index and model, instead
of hard-coding a threshold.

**Calibration is harsh on this catalogue.** The score asks "how does this cosine compare
to the catalogue's real neighbour pairs?" On H&M those pairs are very close (median
0.93), because many products are near-identical basics. So a held-out sneaker whose best
match had cosine 0.83 scored only 0.09. It is a good match, but weaker than most pairs
inside the catalogue. The ranking is unaffected. The number is honest relative to this
catalogue, but it reads as pessimistic to a user.

## 5. Where this started: the original catalogue

The project started on a non-public catalogue (485 product images, many with real
overlay text). There, the label-free checks found the text problem on real data:

| config | near-dup recall | embed 485 imgs |
|---|---|---|
| CLIP ViT-B/32 | 0.44 (102/230) | 6 s |
| CLIP + text masking | 0.85 (196/230) | 162 s |
| DINOv2 S/14 | 0.85 (195/230) | 8 s |

CLIP missed more than half of the *same photo* pairs, because the same photo with a
different banner looked less similar than two unrelated products with similar text.
Masking and DINOv2 both fixed it. The per-image results are not published, because they
name images from that catalogue. The H&M banner trap reproduces the effect with known
answers, and it is smaller there (10% of queries), because a single strip on a clean
studio photo is less text than a full promotional banner.

## 6. What we did not evaluate

- **Human relevance judgements.** The gold standard, and the next step with a real
  product team: a few hundred graded query–result pairs.
- **Online metrics** (click-through, add-to-cart) that would replace all of the above in
  production.
- **A type confusion table** (blouse vs shirt is an honest confusion; bag vs sneakers is
  not) and **cluster purity / ARI against product type**. Both are cheap next steps.
- **Scale.** 998 vectors say nothing about latency at 10 M. See the scale notes in
  [decisions.md](decisions.md).
