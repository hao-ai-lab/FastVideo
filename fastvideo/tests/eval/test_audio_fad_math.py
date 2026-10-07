import numpy as np
import pytest

from fastvideo.eval.metrics.audio.frechet_distance.metric import (
    _frechet_distance, )


def test_frechet_distance_supports_current_scipy_sqrtm_api():
    mean_a = np.array([0.0, 1.0])
    mean_b = np.array([1.0, 3.0])
    covariance = np.array([[2.0, 0.25], [0.25, 1.0]])

    same = _frechet_distance(mean_a, covariance, mean_a, covariance)
    shifted = _frechet_distance(mean_a, covariance, mean_b, covariance)

    assert same == pytest.approx(0.0, abs=1e-10)
    assert shifted == pytest.approx(5.0, abs=1e-10)
