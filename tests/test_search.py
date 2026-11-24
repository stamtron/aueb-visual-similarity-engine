import numpy as np
import pytest
from PIL import Image

from vsim import build_index, search
from vsim.calibration import calibrate, reference_quantiles


def test_search_contract(dataset, tmp_path):
    idx = build_index(dataset, tmp_path / "idx.npz", model="fake")
    res = search(str(dataset / "4.jpg"), 5, idx)
    assert len(res) == 5
    assert res[0].id == 4 and res[0].score == 1.0  # the image itself comes first
    assert all(isinstance(r.id, int) for r in res)
    assert all(0.0 <= r.score <= 1.0 for r in res)
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
    assert all(r.explanation for r in res)
    # 4.jpg is red (ids 1, 4, 7, ...): its neighbours should be red too
    assert all((r.id - 1) % 3 == 0 for r in res)


def test_k_larger_than_index_and_invalid_k(dataset, tmp_path):
    idx = build_index(dataset, tmp_path / "idx.npz", model="fake")
    assert len(search(str(dataset / "1.jpg"), 1000, idx)) == 30
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            search(str(dataset / "1.jpg"), bad, idx)


def test_query_with_different_format_and_size(dataset, tmp_path):
    idx = build_index(dataset, tmp_path / "idx.npz", model="fake")
    q = tmp_path / "q.png"
    Image.open(dataset / "2.jpg").convert("RGBA").resize((300, 200)).save(q)
    assert search(str(q), 1, idx)[0].id == 2


def test_rebuilt_index_is_reloaded(dataset, tmp_path):
    idx = build_index(dataset, tmp_path / "idx.npz", model="fake")
    search(str(dataset / "1.jpg"), 1, idx)
    (dataset / "1.jpg").unlink()
    idx = build_index(dataset, tmp_path / "idx.npz", model="fake")
    assert 1 not in [r.id for r in search(str(dataset / "2.jpg"), 30, idx)]


def test_calibration_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ref = reference_quantiles(emb)
    xs = np.linspace(-1, 1, 201)
    s = calibrate(xs, ref)
    assert s.min() == 0.0 and s.max() == 1.0
    assert np.all(np.diff(s) >= 0)
