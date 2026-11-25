# Documentation

Design notes for the visual similarity engine. Start with
[`../README.md`](../README.md) to install and run it, and read these files to understand
*why* it is built the way it is.

| doc | read it if you want to know… |
|---|---|
| [getting-started.md](getting-started.md) | how to set everything up from scratch on Windows or macOS, in VS Code |
| [usage.md](usage.md) | every CLI command, the Python API, settings, experiments |
| [architecture.md](architecture.md) | how the modules fit together, and what is written to disk |
| [data.md](data.md) | which data we use, how to download it, what folder layout the engine expects |
| [search.md](search.md) | how retrieval, the [0, 1] score, explanations and out-of-distribution detection work |
| [clustering.md](clustering.md) | how K is chosen, how clusters get names, how new images are assigned |
| [evaluation.md](evaluation.md) | how to evaluate without labels (and with them), and what we found |
| [decisions.md](decisions.md) | the trade-offs we made, what we rejected, and what we would do next |

## The problem in one paragraph

You are given a folder of product images. Build two things:

1. **Search.** Given a query image (a URL), return the K most similar images, each with
   a score in [0, 1] and an explanation.
2. **Clustering.** Group the folder into named clusters, then put new images into a
   cluster or call them outliers.

There are no labels and no GPU, and a reviewer has to run it. That last constraint shapes
most of the design: one command to install, no services, tests without downloads, and
every claim backed by an experiment you can re-run.

## How this maps to the lecture

The interview deck ([HTML](../slides/vsim_interview_example.html),
[PDF](../slides/vsim_interview_example.pdf)) uses
this repo as its running take-home example; the careers deck
([HTML](../slides/aueb_data_science_careers.html), [PDF](../slides/aueb_data_science_careers.pdf)) covers the career paths. Each stage of the interview framework has a place in the code:

| framework stage | where it lives here |
|---|---|
| clarify the problem | the contracts in `search.py` and `clustering.py` docstrings |
| inspect the data | `data.md`; the pHash duplicate groups in `evaluation.md` |
| baseline first | CLIP + brute-force cosine, before anything clever |
| evaluate without labels | `experiments/eval_search.py` |
| find the failure mode | the overlay-text trap (`evaluation.md`) |
| fix and measure again | text masking vs DINOv2, with timings |
| ship it | CLI, tests, `.npz` index, README |
