from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from bemppsolver.plotting import VisualizerConfig, generate_plots, load_data
from bemppsolver.postprocess import PrepConfig, prepare_visualization_data


def _write_raw_solver_npz(path: Path) -> None:
    freq_hz = np.array([200.0, 1000.0, 5000.0], dtype=np.float32)
    angles = np.array([-180.0, 0.0, 180.0], dtype=np.float32)
    horizontal = np.array(
        [
            [-12.0, 0.0, -12.0],
            [-10.0, 0.0, -10.0],
            [-8.0, 0.0, -8.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        freq_hz=freq_hz,
        polar_angle_deg=angles,
        horizontal_spl_norm_db=horizontal,
        vertical_spl_norm_db=horizontal,
        impedance_freq_hz=freq_hz,
        impedance_radiator_names=np.array(["HF", "LF"]),
        impedance_real=np.ones((2, 3), dtype=np.float32),
        impedance_imag=np.zeros((2, 3), dtype=np.float32),
    )


def test_prepare_visualization_data_preserves_multiradiator_impedance(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.npz"
    formatted_path = tmp_path / "formatted.npz"
    _write_raw_solver_npz(raw_path)

    prepare_visualization_data(
        PrepConfig(
            input_polar_npz=raw_path,
            output_npz=formatted_path,
            angle_samples=5,
            freq_samples=5,
            octave_smoothing=None,
        )
    )

    with np.load(formatted_path) as data:
        assert data["impedance_real"].shape == (2, 3)
        assert data["impedance_imag"].shape == (2, 3)
        assert data["impedance_radiator_names"].tolist() == ["HF", "LF"]
        assert data["horizontal_isobar_db"].shape == (5, 5)


def test_generate_plots_writes_expected_pngs(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.npz"
    formatted_path = tmp_path / "formatted.npz"
    _write_raw_solver_npz(raw_path)
    prepare_visualization_data(
        PrepConfig(
            input_polar_npz=raw_path,
            output_npz=formatted_path,
            angle_samples=5,
            freq_samples=5,
            octave_smoothing=None,
        )
    )

    outputs = generate_plots(
        load_data(formatted_path),
        VisualizerConfig(
            input_npz=formatted_path,
            output_dir=tmp_path,
            figure_dpi=72,
        ),
    )

    assert set(outputs) == {
        "horizontal_isobar_png",
        "vertical_isobar_png",
        "acoustic_impedance_png",
    }
    assert (tmp_path / "horizontal_isobar.png").exists()
    assert (tmp_path / "vertical_isobar.png").exists()
    assert (tmp_path / "acoustic_impedance.png").exists()
