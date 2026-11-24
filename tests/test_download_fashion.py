"""scripts/download_fashion.py on a tiny synthetic parquet file (no network)."""

import csv
import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_fashion.py"


@pytest.fixture
def dl():
    spec = importlib.util.spec_from_file_location("download_fashion", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jpeg(colour):
    buf = io.BytesIO()
    Image.new("RGB", (60, 80), colour).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def src(tmp_path):
    # 6 Tshirts, 4 Watches, 1 Belt; product 99 has corrupt bytes
    types = ["Tshirts"] * 6 + ["Watches"] * 4 + ["Belts"]
    ids = [10 + i for i in range(len(types))]
    ids[0] = 99
    rows = {"id": ids, "articleType": types}
    for c in ["subCategory", "masterCategory", "baseColour", "gender", "usage", "season",
              "productDisplayName"]:
        rows[c] = ["x"] * len(ids)
    rows["image"] = [{"bytes": b"not a jpeg" if i == 99 else _jpeg((i * 20 % 255, 0, 0)),
                      "path": None} for i in ids]
    d = tmp_path / "raw" / "data"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict(rows), d / "train-0.parquet")
    return tmp_path / "raw"


def test_stratified_is_balanced_and_deterministic(dl, src):
    rows = dl.read_labels(sorted(src.glob("**/*.parquet")))
    a = dl.stratified(rows, n_types=5, per_class=3, seed=0)
    b = dl.stratified(rows, n_types=5, per_class=3, seed=0)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    # Belts has only 1 image, so it is dropped rather than under-filled
    assert sorted({r["articleType"] for r in a}) == ["Tshirts", "Watches"]
    assert len(a) == 6


def test_main_writes_images_and_labels(dl, src, tmp_path):
    out, labels = tmp_path / "images", tmp_path / "labels.csv"
    args = ["--src", str(src), "--out", str(out), "--labels", str(labels),
            "--types", "1", "--per-class", "6"]
    assert dl.main(args) == 0
    written = sorted(int(p.stem) for p in out.glob("*.jpg"))
    assert 99 not in written and len(written) == 5  # corrupt bytes are skipped
    with open(labels, newline="") as f:
        assert sorted(int(r["id"]) for r in csv.DictReader(f)) == written
    assert not list(out.glob("*.part"))
    assert dl.main(args) == 0  # re-running is a no-op
    assert sorted(int(p.stem) for p in out.glob("*.jpg")) == written
