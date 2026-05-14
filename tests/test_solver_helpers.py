import numpy as np

from bemppsolver.config import CrossoverConfig
from bemppsolver.solver import HornBEMSolver, _split_frequencies_evenly


def test_split_frequencies_evenly_preserves_all_points() -> None:
    freqs = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    chunks = _split_frequencies_evenly(freqs, worker_count=3)

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert np.concatenate(chunks).tolist() == freqs.tolist()


def test_linkwitz_riley_response_is_complex_and_bounded() -> None:
    crossover = CrossoverConfig(
        type="lowpass",
        filter="linkwitz_riley",
        order=4,
        frequency_hz=1200.0,
    )

    response = HornBEMSolver._crossover_response(None, crossover, 1200.0)

    assert isinstance(response, complex)
    assert 0.0 < abs(response) <= 1.0

