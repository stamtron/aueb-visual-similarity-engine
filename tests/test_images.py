import numpy as np
import pytest
from PIL import Image

from vsim.images import ImageLoadError, list_images, load_image


def test_ids_sorted_numerically(tmp_path):
    for name in ["10.jpg", "2.jpg", "1.jpg", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    assert [f.id for f in list_images(tmp_path)] == [1, 2, 10]


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "1.jpg").write_bytes(b"x")
    (tmp_path / "1.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="Duplicate"):
        list_images(tmp_path)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "CMYK", "LA"])
def test_any_mode_becomes_rgb(tmp_path, mode):
    p = tmp_path / ("x.png" if mode != "CMYK" else "x.jpg")
    Image.new(mode, (40, 30)).save(p)
    assert load_image(p).mode == "RGB"


def test_transparent_png_composited_on_white(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(p)
    assert np.asarray(load_image(p)).min() == 255


def test_large_image_downscaled(tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (4000, 1000)).save(p)
    assert max(load_image(p).size) == 1024


def test_garbage_raises_clean_error(tmp_path):
    p = tmp_path / "fake.jpg"
    p.write_text("<html>not an image</html>")
    with pytest.raises(ImageLoadError):
        load_image(p)
    with pytest.raises(ImageLoadError):
        load_image("ftp://example.com/x.jpg")
