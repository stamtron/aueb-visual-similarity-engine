"""scripts/download_hm.py on a tiny fake articles.csv + images folder (no network)."""

import csv
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_hm.py"


@pytest.fixture
def dl():
    spec = importlib.util.spec_from_file_location("download_hm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def raw(tmp_path, dl):
    """4 Sweater products (one with 3 colour variants), 2 Socks, 1 Belt.
    Article 0100000006 has no photo, 0100000007 has corrupt bytes."""
    rows = [  # (article_id, product_code, type)
        ("0100000001", "1", "Sweater"), ("0100000002", "1", "Sweater"),
        ("0100000003", "1", "Sweater"), ("0100000004", "2", "Sweater"),
        ("0100000005", "3", "Sweater"), ("0100000007", "4", "Sweater"),
        ("0200000001", "5", "Socks"), ("0100000006", "6", "Socks"),
        ("0300000001", "7", "Belt"),
    ]
    art = tmp_path / "articles.csv"
    with open(art, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["article_id", *dl.LABEL_COLS])
        w.writeheader()
        for aid, code, typ in rows:
            w.writerow({"article_id": aid, "product_code": code, "product_type_name": typ,
                        **{c: "x" for c in dl.LABEL_COLS[2:] if c != "product_type_name"}})
    images = tmp_path / "images"
    for aid, _, _ in rows:
        if aid == "0100000006":
            continue
        d = images / aid[:3]
        d.mkdir(parents=True, exist_ok=True)
        if aid == "0100000007":
            (d / f"{aid}.jpg").write_bytes(b"not a jpeg")
        else:
            Image.new("RGB", (1000, 1500), (int(aid[-1]) * 30, 0, 0)).save(d / f"{aid}.jpg")
    return art, images


def test_stratified_one_variant_per_product_and_deterministic(dl, raw):
    rows = dl.read_articles(raw[0])
    a = dl.stratified(rows, n_types=5, per_class=2, seed=0)
    assert [r["article_id"] for r in a] == [r["article_id"]
                                            for r in dl.stratified(rows, 5, 2, 0)]
    assert sorted({r["product_type_name"] for r in a}) == ["Socks", "Sweater"]  # Belt: too few
    codes = [r["product_code"] for r in a]
    assert len(codes) == len(set(codes)) == 4


def test_main_from_folder_writes_small_images_and_labels(dl, raw, tmp_path):
    art, images = raw
    out, labels = tmp_path / "out", tmp_path / "labels.csv"
    args = ["--articles", str(art), "--images-dir", str(images), "--out", str(out),
            "--labels", str(labels), "--types", "1", "--per-class", "4"]
    assert dl.main(args) == 0
    written = sorted(int(p.stem) for p in out.glob("*.jpg"))
    assert 100000007 not in written and len(written) == 3  # corrupt bytes are skipped
    assert max(Image.open(out / f"{written[0]}.jpg").size) == dl.MAX_SIDE
    with open(labels, newline="") as f:
        assert sorted(int(r["article_id"]) for r in csv.DictReader(f)) == written
    assert not list(out.glob("*.part"))
    assert dl.main(args) == 0  # re-running is a no-op
    assert sorted(int(p.stem) for p in out.glob("*.jpg")) == written


@pytest.mark.parametrize("layout", ["flat", "nested", "zipped"])
def test_kaggle_source_finds_the_file_wherever_kaggle_puts_it(dl, tmp_path, layout):
    """kaggle 1.x saves flat, 2.x keeps images/058/, and both may zip single files."""
    import zipfile

    aid = "0588119001"

    class FakeApi:
        def competition_download_file(self, comp, name, path, quiet):
            dest = Path(path) / (name if layout != "flat" else Path(name).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if layout == "zipped":
                with zipfile.ZipFile(dest.with_name(dest.name + ".zip"), "w") as z:
                    z.writestr(dest.name, b"jpeg bytes")
            else:
                dest.write_bytes(b"jpeg bytes")

    src = dl.KaggleSource.__new__(dl.KaggleSource)  # skip authentication
    src.api, src.tmp = FakeApi(), tmp_path / "dl"
    src.tmp.mkdir()
    assert src.get(aid) == b"jpeg bytes"
    assert not [p for p in src.tmp.rglob("*") if p.is_file()]  # cleaned up


def test_main_unzips_a_hand_downloaded_articles_zip(dl, raw, tmp_path):
    """`kaggle ... -f articles.csv` gives articles.csv.zip; no unzip tool needed."""
    import zipfile

    art, images = raw
    z = tmp_path / "hm" / "articles.csv.zip"
    z.parent.mkdir()
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(art, "articles.csv")
    args = ["--articles", str(z.parent / "articles.csv"), "--images-dir", str(images),
            "--out", str(tmp_path / "out"), "--labels", str(tmp_path / "labels.csv"),
            "--types", "1", "--per-class", "4"]
    assert dl.main(args) == 0
    assert (z.parent / "articles.csv").read_bytes() == art.read_bytes()
