import numpy as np
import pytest
import torch

from fastvideo.models.dits.hyworld.retrieval_context import (
    generate_points_in_sphere,
    make_retrieval_generator,
    select_aligned_memory_frames,
)


def _w2c_at_x(x: float) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[0, 3] = x
    return np.linalg.inv(c2w)


def _symmetric_camera_path() -> np.ndarray:
    poses = np.stack([_w2c_at_x(0.0) for _ in range(36)])
    poses[4:8] = _w2c_at_x(1.0)
    poses[8:12] = _w2c_at_x(-1.0)
    poses[12:16] = _w2c_at_x(4.0)
    return poses


def _select_memory(points: torch.Tensor) -> list[int]:
    return select_aligned_memory_frames(
        _symmetric_camera_path(),
        current_frame_idx=28,
        memory_frames=20,
        temporal_context_size=12,
        pred_latent_size=4,
        device="cpu",
        points_local=points,
    )


def test_explicit_seed_is_independent_of_global_rng() -> None:
    torch.manual_seed(11)
    torch.rand(37)
    points_a = generate_points_in_sphere(4096, 8.0, generator=make_retrieval_generator(seed=1234))

    torch.manual_seed(99)
    torch.rand(113)
    points_b = generate_points_in_sphere(4096, 8.0, generator=make_retrieval_generator(seed=1234))

    torch.testing.assert_close(points_a, points_b, rtol=0, atol=0)
    assert _select_memory(points_a) == _select_memory(points_b)


def test_retrieval_does_not_advance_source_or_global_rng() -> None:
    source = torch.Generator(device="cpu").manual_seed(4321)
    torch.rand(17, generator=source)
    source_state = source.get_state().clone()

    torch.manual_seed(8765)
    global_state = torch.random.get_rng_state().clone()

    points = generate_points_in_sphere(256, 2.0, generator=make_retrieval_generator(source))

    assert points.shape == (256, 3)
    assert torch.equal(source.get_state(), source_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_same_initial_seed_ignores_source_state_progress() -> None:
    first = torch.Generator(device="cpu").manual_seed(31415)
    second = torch.Generator(device="cpu").manual_seed(31415)
    torch.rand(13, generator=first)
    torch.rand(79, generator=second)

    points_first = generate_points_in_sphere(512, 3.0, generator=make_retrieval_generator(first))
    points_second = generate_points_in_sphere(512, 3.0, generator=make_retrieval_generator(second))

    torch.testing.assert_close(points_first, points_second, rtol=0, atol=0)


def test_source_generator_takes_precedence_over_seed() -> None:
    source = torch.Generator(device="cpu").manual_seed(17)

    from_source = generate_points_in_sphere(128, 1.0, generator=make_retrieval_generator(source, seed=999))
    expected = generate_points_in_sphere(128, 1.0, generator=make_retrieval_generator(seed=17))

    torch.testing.assert_close(from_source, expected, rtol=0, atol=0)


def test_generator_list_uses_first_trajectory_seed() -> None:
    first = torch.Generator(device="cpu").manual_seed(7)
    second = torch.Generator(device="cpu").manual_seed(9)

    from_list = generate_points_in_sphere(128, 1.0, generator=make_retrieval_generator([first, second]))
    from_first = generate_points_in_sphere(128, 1.0, generator=make_retrieval_generator(first))

    torch.testing.assert_close(from_list, from_first, rtol=0, atol=0)


def test_global_seed_fallback_is_reproducible_without_advancing_rng() -> None:
    torch.manual_seed(2468)
    global_state = torch.random.get_rng_state().clone()

    fallback_points = generate_points_in_sphere(128, 1.0, generator=make_retrieval_generator())
    explicit_points = generate_points_in_sphere(128, 1.0, generator=make_retrieval_generator(seed=2468))

    torch.testing.assert_close(fallback_points, explicit_points, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_known_seeds_can_change_selected_symmetric_chunk() -> None:
    points_zero = generate_points_in_sphere(4096, 8.0, generator=make_retrieval_generator(seed=0))
    points_two = generate_points_in_sphere(4096, 8.0, generator=make_retrieval_generator(seed=2))

    assert _select_memory(points_zero)[4:8] == [4, 5, 6, 7]
    assert _select_memory(points_two)[4:8] == [8, 9, 10, 11]


def test_invalid_generator_sequences_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_retrieval_generator([])
    with pytest.raises(TypeError, match="torch.Generator"):
        make_retrieval_generator("not-a-generator")
