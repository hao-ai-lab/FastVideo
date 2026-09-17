# SPDX-License-Identifier: Apache-2.0
from PIL import Image

from fastvideo_studio import thumbnails


def test_image_poster_is_small_and_invalidates_after_source_change(tmp_path):
    source = tmp_path / "output.png"
    cache = tmp_path / "posters"
    Image.new("RGB", (832, 480), "red").save(source)
    first = thumbnails.get_thumbnail(source, cache)
    with Image.open(first) as poster:
        assert poster.format == "JPEG"
        assert poster.width == 320
        assert poster.height < 320
    assert thumbnails.get_thumbnail(source, cache) == first
    Image.new("RGB", (480, 832), "blue").save(source)
    second = thumbnails.get_thumbnail(source, cache)
    assert second != first
    with Image.open(second) as poster:
        assert poster.height == 320


def test_poster_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(thumbnails, "MAX_CACHED_POSTERS", 2)
    cache = tmp_path / "posters"
    for index in range(3):
        source = tmp_path / f"{index}.png"
        Image.new("RGB", (10, 10)).save(source)
        latest = thumbnails.get_thumbnail(source, cache)
    assert len(list(cache.glob("*.jpg"))) == 2
    assert latest.exists()
