import json

import numpy as np
from PIL import Image

from vsim import assign, preprocess
from vsim.clustering import OUTLIER, load_dataset


def test_preprocess_is_cached_and_content_addressed(dataset):
    a = preprocess(dataset, model="fake")
    assert preprocess(dataset, model="fake") == a
    Image.new("RGB", (64, 48), (200, 30, 30)).save(dataset / "1.jpg")
    assert preprocess(dataset, model="fake") != a


def test_clusters_recover_colour_families(dataset):
    ds = load_dataset(preprocess(dataset, model="fake", k=3))
    groups = {}
    for i, c in zip(ds["ids"], ds["assign"]):
        groups.setdefault(int(c), set()).add((int(i) - 1) % 3)
    assert all(len(g) == 1 for g in groups.values())
    assert len(set(map(str, ds["labels"]))) == 3  # labels are unique


def test_assign_member_like_image_and_outlier(dataset, tmp_path):
    ds_id = preprocess(dataset, model="fake", k=3)
    ds = load_dataset(ds_id)
    red_cluster = int(ds["assign"][list(map(str, ds["ids"])).index("1")])

    q = tmp_path / "red.png"
    arr = np.clip(np.array([200, 30, 30]) + np.random.default_rng(5).normal(0, 12, (48, 64, 3)),
                  0, 255).astype(np.uint8)
    Image.fromarray(arr).save(q)
    r = assign(str(q), ds_id)
    assert r.cluster == red_cluster and not r.is_outlier

    odd = tmp_path / "odd.png"
    Image.fromarray(np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))).save(odd)
    assert assign(str(odd), ds_id).label == OUTLIER


def test_assign_is_idempotent(dataset, tmp_path):
    ds_id = preprocess(dataset, model="fake", k=3)
    q = tmp_path / "q.png"
    Image.open(dataset / "5.jpg").resize((50, 40)).save(q)
    first, second = assign(str(q), ds_id), assign(str(q), ds_id)
    assert first.label == second.label and second.already_known
    lines = (load_dataset(ds_id)["dir"] / "additions.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["assignment"]["label"] == first.label


def test_dataset_member_returns_its_cluster_without_writing(dataset):
    ds_id = preprocess(dataset, model="fake", k=3)
    r = assign(str(dataset / "7.jpg"), ds_id)
    assert r.already_known and r.nearest_ids == [7]
    assert not (load_dataset(ds_id)["dir"] / "additions.jsonl").exists()


def test_labels_name_a_second_product_only_when_it_is_really_there():
    from vsim import tags
    from vsim.clustering import make_labels

    col = {n: i for i, n in enumerate(tags.TAG_NAMES)}
    probs = np.full((6, len(tags.TAG_NAMES)), 0.01, dtype=np.float32)
    probs[:3, col["garment:watch"]] = 0.9                          # cluster 0: only watches
    probs[3:, col["garment:handbag"]] = [0.8, 0.1, 0.8]            # cluster 1: bags and
    probs[3:, col["garment:wallet"]] = [0.1, 0.8, 0.1]             # wallets, 2:1
    labels = make_labels(probs, np.array([0, 0, 0, 1, 1, 1]))
    assert labels[0].endswith(": watch")
    assert labels[1].endswith(": handbag & wallet")
