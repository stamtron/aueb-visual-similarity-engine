# Evaluation

"It looks right on five queries" is not an evaluation. This page explains how we measure
search quality **without relevance labels**, what we found, and how labels change the
picture once we have them.

Everything here comes from one script you can re-run:

```bash
HF_HUB_OFFLINE=1 uv run python experiments/eval_search.py data/images
# -> experiments/results/search_eval.json
```

> **Status of the numbers.** The table below was produced on the original, non-public
> catalogue this project started from (485 product images, many with overlay text).
> It is kept because the *method* and the *failure mode* are the point. The per-image
> results file is not published, because it names images from that catalogue. The
> numbers will be regenerated on the public H&M sample ([data.md](data.md)), and the
> results file committed, then this page updated.

## 1. Weak ground truth when there are no labels

We build three kinds of "free" ground truth from the images themselves.

### Near-duplicate recall

Catalogues reuse the same photo with a different crop, banner or language. A perceptual
hash (pHash, Hamming distance ≤ 14) finds these groups. They are unambiguous positives:
whatever "similar" means, the same photo must count.

For an image in a group of size *g*, we ask: are its *g − 1* mates in its top *g − 1*
results? Summed over all groups, that gives 230 mate slots on the original data.

pHash is a *weak* labeller. It only finds near-identical pixels, so a high score here is
necessary, not sufficient.

### Augmentation recall@1

Take an indexed image, transform it, and query with the copy. The original should be
rank 1. Transforms: greyscale, downscale to 128 px, JPEG quality 10, horizontal mirror,
70% centre crop. This tests robustness to things a user's upload will actually do.

### Trap queries

Hand-picked queries whose same-photo mates carry **different overlay text**. They exist
to test one hypothesis: *is the model matching the words on the image instead of the
product?* We report the rank of each mate.

### Sanity checks

- **Out-of-distribution:** random noise, a solid blue square, black text on white. Each
  should be flagged by the nn1 threshold ([search.md](search.md)).
- **Timing:** embedding time for the whole folder and per-query search latency, on a
  laptop CPU.

## 2. Results (original data, CPU)

| config | near-dup recall | aug R@1 (mirror / crop) | trap mates found in top 3 | OOD all flagged | embed 485 imgs |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | **0.443** (102/230) | 0.86 / 0.82 | 1 of 5 | yes | 6.1 s |
| CLIP + text masking | 0.852 (196/230) | 0.96 / 0.78 | 5 of 5 | **no** (solid blue) | 162.1 s |
| DINOv2 S/14 | 0.848 (195/230) | 1.00 / 0.86 | 5 of 5 | yes | 8.2 s |
| DINOv2 + text masking | 0.848 (195/230) | 0.94 / 0.86 | 5 of 5 | yes | 160.9 s |

Search latency was 4–5 µs per query for every configuration (brute-force cosine over 485
vectors). Greyscale, downscale and JPEG recall@1 were ≥ 0.88 everywhere. Re-running the
script writes the full numbers to `experiments/results/search_eval.json`.

## 3. What we learned

**The baseline had a real failure mode, and only the label-free checks exposed it.**
CLIP's near-duplicate recall was 0.44: it missed more than half of the *same photo*
pairs. The trap queries explain why. For one trap, CLIP ranked the two same-photo mates
7th and 13th, behind unrelated products that carried *similar text*. CLIP was trained on
image–caption pairs, so it reads. On a catalogue full of banners, it matches the banner.

**Two fixes, very different costs.**

- *Mask the text, keep CLIP.* EasyOCR + inpainting brings recall to 0.85 and fixes the
  traps. But embedding gets **~27× slower** (6 s → 162 s), it adds a heavy optional
  dependency, and the inpainted solid-blue square now looks enough like a product
  (cos 0.774 vs threshold 0.773) that it is **no longer flagged** as out-of-distribution.
- *Switch to DINOv2.* A self-supervised model trained on pixels, not captions. Same
  recall (0.85), best augmentation robustness, all OOD probes flagged, and only 8 s to
  embed. Masking on top of DINOv2 adds nothing.

**Why CLIP is still the default.** DINOv2 wins on "same photo". CLIP is what gives us
zero-shot **tags** (pattern, colour, garment), which drive the explanations and the
cluster names. The decision is a trade-off, written down in [decisions.md](decisions.md),
and switching is one flag: `vsim index … --model dinov2-s14`.

**Raw cosine is not a score.** CLIP nn1 cosines sit in a narrow band (2nd percentile
0.717, median 0.857), and random noise still reaches 0.557. DINOv2's band is very
different (p2 0.456, median 0.688). This is why we calibrate per index and model,
instead of hard-coding a threshold.

## 4. With labels: precision@K

The H&M sample has clean catalogue labels (product type, colour, graphical appearance,
garment group). That enables proper metrics:

- **precision@K by product type:** for each image, what fraction of its top K results
  share its `product_type_name`? Macro-averaged over types (the sample is balanced, 50 per type).
- **precision@K by colour**, to see whether a model groups by *what* or by *colour*.
- **a confusion table** between types (blouse vs shirt is an honest confusion;
  bag vs sneakers is not).
- **cluster purity / ARI vs product type**, for the clustering side.

Caveat: "same product type" is only one notion of similarity. Two black sweaters share a
label with a pink one, and a shopper may want the black one first.

### Reproducing the overlay-text trap

These photos carry almost no text, so the failure mode is built synthetically, which has
the advantage of known answers:

- **same photo, different banner** ("SALE -50%" vs "NEW IN"): should still match.
- **different product, same banner**: should *not* match because of the banner.

This replaces the hand-picked trap ids from the original data. This is planned work.

## 5. What we did not evaluate

- **Human relevance judgements.** The gold standard, and the next step with a real
  product team: a few hundred graded query–result pairs.
- **Online metrics** (click-through, add-to-cart) that would replace all of the above in
  production.
- **Scale.** 485 vectors say nothing about latency at 10 M. See the scale notes in
  [decisions.md](decisions.md).
