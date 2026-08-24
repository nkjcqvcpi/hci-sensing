"""HDF5 and manifest I/O without assumptions about private file naming."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import ProjectConfig
from .signal import preprocess_cir

CANONICAL_CHANNELS = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class ManifestRow:
    recording_path: str
    participant_id: str
    recording_id: str
    site: str
    caliper_mm: float


def read_manifest(path: str | Path) -> list[ManifestRow]:
    required = {"recording_path", "participant_id", "recording_id", "site", "caliper_mm"}
    rows: list[ManifestRow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for raw in reader:
            rows.append(
                ManifestRow(
                    recording_path=raw["recording_path"],
                    participant_id=raw["participant_id"],
                    recording_id=raw["recording_id"],
                    site=raw["site"].strip().lower(),
                    caliper_mm=float(raw["caliper_mm"]),
                )
            )
    if not rows:
        raise ValueError("Manifest contains no recordings")
    return rows


def _to_complex(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return array
    if array.dtype.fields:
        names = {name.lower(): name for name in array.dtype.fields}
        if "r" in names and "i" in names:
            return array[names["r"]] + 1j * array[names["i"]]
        if "real" in names and "imag" in names:
            return array[names["real"]] + 1j * array[names["imag"]]
    if array.shape[-1:] == (2,):
        return array[..., 0] + 1j * array[..., 1]
    raise TypeError("Dataset is real-valued and has no recognizable real/imaginary representation")


def canonicalize_cir(array: np.ndarray, layout: str = "auto") -> np.ndarray:
    """Return complex CIR with shape [frames, 4 channels, bins]."""
    array = _to_complex(array)
    if array.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional CIR after complex decoding, got {array.shape}"
        )

    layouts = {
        "frames_channels_bins": (0, 1, 2),
        "channels_bins_frames": (2, 0, 1),
        "bins_channels_frames": (2, 1, 0),
    }
    if layout != "auto":
        if layout not in layouts:
            raise ValueError(f"Unknown layout {layout!r}; choose one of {sorted(layouts)}")
        result = np.transpose(array, layouts[layout])
    else:
        candidates: list[np.ndarray] = []
        for axes in layouts.values():
            candidate = np.transpose(array, axes)
            if candidate.shape[1] == 4 and candidate.shape[0] > 4 and candidate.shape[2] > 4:
                candidates.append(candidate)
        likely = [
            item for item in candidates if item.shape[0] >= item.shape[2] and item.shape[2] <= 512
        ]
        if len(likely) == 1:
            result = likely[0]
        elif len(candidates) == 1:
            result = candidates[0]
        else:
            raise ValueError(
                f"Cannot uniquely infer CIR layout from {array.shape}; pass --layout explicitly"
            )
    return result.astype(np.complex64, copy=False)


def _channel_tuple(name: str) -> tuple[int, int] | None:
    tail = name.rsplit("/", 1)[-1].split("_", 1)[-1]
    try:
        value = ast.literal_eval(tail)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, tuple) or len(value) not in {2, 3}:
        return None
    tx, rx = value[-2:]
    if tx in {0, 1} and rx in {0, 1}:
        return int(tx), int(rx)
    return None


def _tuple_datasets(handle: h5py.File) -> dict[tuple[int, int], np.ndarray]:
    found: dict[tuple[int, int], np.ndarray] = {}

    def visitor(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            channel = _channel_tuple(name)
            if channel is not None:
                found[channel] = _to_complex(np.asarray(obj))

    handle.visititems(visitor)
    return found


def load_hdf5_recording(
    path: str | Path,
    dataset: str | None = None,
    layout: str = "auto",
) -> np.ndarray:
    """Load either a shared `/cir` dataset or four tuple-named channel datasets."""
    with h5py.File(path, "r") as handle:
        if dataset is not None:
            if dataset not in handle:
                raise KeyError(f"Dataset {dataset!r} not found in {path}")
            return canonicalize_cir(np.asarray(handle[dataset]), layout)

        channels = _tuple_datasets(handle)
        if set(channels) == set(CANONICAL_CHANNELS):
            normalized: list[np.ndarray] = []
            for channel in CANONICAL_CHANNELS:
                value = np.squeeze(channels[channel])
                if value.ndim != 2:
                    raise ValueError(f"Channel {channel} must be [frames, bins], got {value.shape}")
                if value.shape[0] < value.shape[1] and value.shape[0] <= 256:
                    value = value.T
                normalized.append(value)
            shapes = {item.shape for item in normalized}
            if len(shapes) != 1:
                raise ValueError(f"Tuple channel shapes disagree: {sorted(shapes)}")
            return np.stack(normalized, axis=1).astype(np.complex64)

        if "/cir" in handle:
            return canonicalize_cir(np.asarray(handle["/cir"]), layout)
        raise ValueError(
            "No complete set of tuple-named datasets and no /cir dataset found. "
            "Pass --dataset for the intended numeric dataset."
        )


def build_observation_archive(
    manifest_path: str | Path,
    output_path: str | Path,
    config: ProjectConfig,
    dataset: str | None = None,
    layout: str = "auto",
) -> None:
    """Preprocess recordings into a metadata-preserving NPZ archive."""
    responses: list[np.ndarray] = []
    participant_ids: list[str] = []
    recording_ids: list[str] = []
    sites: list[str] = []
    calipers: list[float] = []
    frequencies: np.ndarray | None = None

    for row in read_manifest(manifest_path):
        cir = load_hdf5_recording(row.recording_path, dataset=dataset, layout=layout)
        result = preprocess_cir(cir, config.acquisition)
        n_windows = result.response.shape[0]
        responses.append(result.response)
        participant_ids.extend([row.participant_id] * n_windows)
        recording_ids.extend([row.recording_id] * n_windows)
        sites.extend([row.site] * n_windows)
        calipers.extend([row.caliper_mm] * n_windows)
        if frequencies is None:
            frequencies = result.frequencies_hz
        elif not np.allclose(frequencies, result.frequencies_hz):
            raise ValueError("Recordings produced inconsistent frequency grids")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        response=np.concatenate(responses).astype(np.complex64),
        frequencies_hz=frequencies,
        participant_id=np.asarray(participant_ids, dtype="U"),
        recording_id=np.asarray(recording_ids, dtype="U"),
        site=np.asarray(sites, dtype="U"),
        caliper_mm=np.asarray(calipers, dtype=np.float32),
    )
