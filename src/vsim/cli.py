"""Command line: `vsim index|search|cluster|assign|clusters`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .models import DEFAULT_MODEL, MODEL_ALIASES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vsim", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_opts(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"alias ({', '.join(MODEL_ALIASES)}) or full spec")
        sp.add_argument("--text-mask", action="store_true",
                        help="remove overlay text with OCR + inpainting before embedding")

    sp = sub.add_parser("index", help="build a search index from a folder")
    sp.add_argument("folder")
    sp.add_argument("--out", default="index.npz")
    model_opts(sp)

    sp = sub.add_parser("search", help="find the K most similar images to a URL or path")
    sp.add_argument("url")
    sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--index", default="index.npz")

    sp = sub.add_parser("cluster", help="cluster a folder, print the dataset id")
    sp.add_argument("folder")
    sp.add_argument("--k", type=int, default=None, help="force K instead of choosing it")
    model_opts(sp)

    sp = sub.add_parser("assign", help="assign an image (URL or path) to a cluster")
    sp.add_argument("url")
    sp.add_argument("dataset_id")
    sp.add_argument("--dry-run", action="store_true", help="do not persist the assignment")

    sp = sub.add_parser("clusters", help="show the clusters of a dataset id")
    sp.add_argument("dataset_id")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if a.cmd == "index":
            from .search import build_index
            print(build_index(a.folder, a.out, a.model, a.text_mask))
        elif a.cmd == "search":
            from .search import search
            for r in search(a.url, a.k, a.index):
                print(f"{r.id!s:>6}  {r.score:.3f}  {r.explanation}")
        elif a.cmd == "cluster":
            from .clustering import preprocess
            print(preprocess(a.folder, a.model, a.text_mask, k=a.k))
        elif a.cmd == "assign":
            from .clustering import assign
            r = assign(a.url, a.dataset_id, persist=not a.dry_run)
            print(r.label)
            print(f"  best cosine {r.similarity:.3f} (outlier below {r.threshold:.3f}); "
                  f"nearest: {r.nearest_ids}" + ("; already known" if r.already_known else ""),
                  file=sys.stderr)
        elif a.cmd == "clusters":
            from .clustering import load_dataset
            s = load_dataset(a.dataset_id)["summary"]
            for c in s["clusters"]:
                print(f"[{c['cluster']}] {c['label']}  (n={c['size']})  e.g. {c['members'][:8]}")
            if s["k_selection"]:
                print("\nK selection:", json.dumps(s["k_selection"]))
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
