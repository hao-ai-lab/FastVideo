import os

from fastvideo.entrypoints.video_generator import VideoGenerator


def _new_video_generator() -> VideoGenerator:
    # Bypass __init__ since we only test a pure helper method.
    return VideoGenerator.__new__(VideoGenerator)


def test_prepare_output_path_file_sanitization(tmp_path):
    vg = _new_video_generator()
    target_dir = tmp_path / "dir"
    raw_path = target_dir / "inv:al*id?.mp4"

    result = vg._prepare_output_path(str(raw_path), prompt="ignored")

    assert os.path.dirname(result) == str(target_dir)
    assert os.path.basename(result) == "invalid.mp4"
    assert os.path.isdir(target_dir)


def test_prepare_output_path_directory_prompt_derived(tmp_path):
    vg = _new_video_generator()
    out_dir = tmp_path / "outputs"
    prompt = "Hello:/\\*?<>| world"

    result = vg._prepare_output_path(str(out_dir), prompt=prompt)

    assert os.path.dirname(result) == str(out_dir)
    # spaces are preserved (collapsed) by sanitizer; here it becomes "Hello world.mp4"
    assert os.path.basename(result) == "Hello world.mp4"
    assert os.path.isdir(out_dir)


def test_prepare_output_path_non_mp4_treated_as_dir(tmp_path):
    vg = _new_video_generator()
    weird_dir = tmp_path / "foo.gif"
    prompt = "My Video"

    result = vg._prepare_output_path(str(weird_dir), prompt=prompt)

    assert os.path.dirname(result) == str(weird_dir)
    assert os.path.basename(result) == "My Video.mp4"
    assert os.path.isdir(weird_dir)


def test_prepare_output_path_uniqueness_suffix(tmp_path):
    vg = _new_video_generator()
    out_dir = tmp_path / "outputs"
    prompt = "Sample Name"

    first = vg._prepare_output_path(str(out_dir), prompt=prompt)
    # simulate existing file
    os.makedirs(os.path.dirname(first), exist_ok=True)
    with open(first, "wb") as f:
        f.write(b"")

    second = vg._prepare_output_path(str(out_dir), prompt=prompt)
    assert os.path.basename(second) == "Sample Name_1.mp4"

    # simulate second existing file as well
    with open(second, "wb") as f:
        f.write(b"")
    third = vg._prepare_output_path(str(out_dir), prompt=prompt)
    assert os.path.basename(third) == "Sample Name_2.mp4"


def test_prepare_output_path_empty_prompt_fallback(tmp_path):
    vg = _new_video_generator()
    out_dir = tmp_path / "outputs"
    bad_prompt = ":/\\*?<>|   .."  # sanitizes to empty, should fallback to "video"

    result = vg._prepare_output_path(str(out_dir), prompt=bad_prompt)

    assert os.path.dirname(result) == str(out_dir)
    assert os.path.basename(result) == "video.mp4"

