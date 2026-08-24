"""Deterministic synthetic frames for smoke tests and API demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FRAME_COLUMN, LABEL_COLUMN, SESSION_COLUMN


def make_synthetic_frame_table(
    *,
    class_count: int = 19,
    sessions_per_class: int = 3,
    frames_per_session: int = 20,
    random_state: int = 7,
) -> pd.DataFrame:
    if not 2 <= class_count <= 19:
        raise ValueError("class_count must be in [2, 19]")
    if sessions_per_class < 1 or frames_per_session < 5:
        raise ValueError("Synthetic recordings require at least one session and five frames")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    for label in range(class_count):
        for session_index in range(sessions_per_class):
            session = f"class-{label:02d}-session-{session_index:02d}"
            phase = label * 0.17 + session_index * 0.02
            for frame_index in range(frames_per_session):
                time = frame_index / max(frames_per_session - 1, 1)
                real_0 = np.cos(phase + time) + rng.normal(0, 0.015)
                imag_0 = np.sin(phase + time) + rng.normal(0, 0.015)
                real_1 = np.cos(phase * 1.7 + time) + rng.normal(0, 0.015)
                imag_1 = np.sin(phase * 1.7 + time) + rng.normal(0, 0.015)
                rows.append(
                    {
                        "distance": 90.0 + 2.5 * label + rng.normal(0, 0.2),
                        "azimuth": -20.0 + 2.0 * label + rng.normal(0, 0.15),
                        "azimuth_fom": 95.0 - label * 0.3 + rng.normal(0, 0.1),
                        "elevation": 5.0 + label * 0.8 + rng.normal(0, 0.1),
                        "pdoa1": label * 1.5 + rng.normal(0, 0.1),
                        "i_nlos_1": int(label % 4 == 0),
                        "i_first_path_idx_1": 12.0 + label * 0.1 + rng.normal(0, 0.02),
                        "i_main_path_idx_1": 15.0 + label * 0.1 + rng.normal(0, 0.02),
                        "i_snr_main_path_1": 30.0 + label + rng.normal(0, 0.2),
                        "i_rssi_1": -70.0 + label * 0.7 + rng.normal(0, 0.2),
                        "i_noise_variance_1": 3.0 + label * 0.05 + rng.normal(0, 0.01),
                        "i_cir_re_1_0": real_0,
                        "i_cir_im_1_0": imag_0,
                        "i_cir_re_1_1": real_1,
                        "i_cir_im_1_1": imag_1,
                        SESSION_COLUMN: session,
                        FRAME_COLUMN: frame_index,
                        LABEL_COLUMN: label,
                    }
                )
    return pd.DataFrame(rows)


def write_synthetic_csvs(output_directory: str | Path, **kwargs: int) -> list[Path]:
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    table = make_synthetic_frame_table(**kwargs)
    paths: list[Path] = []
    for session, recording in table.groupby(SESSION_COLUMN, sort=True):
        output = recording.drop(columns=[SESSION_COLUMN]).rename(columns={LABEL_COLUMN: "Posture"})
        path = destination / f"{session}.csv"
        output.to_csv(path, index=False)
        paths.append(path)
    return paths
