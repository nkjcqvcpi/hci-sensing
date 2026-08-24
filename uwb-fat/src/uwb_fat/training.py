"""Leakage-safe training of one outer LOSO fold."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .data import ObservationArchive, ObservationDataset
from .inference import grid_search
from .loss import physics_informed_loss
from .metrics import aggregate_by_recording, regression_metrics
from .model import UWBFatModel, trainable_parameter_count
from .signal import ComplexStandardizer
from .splits import full_loso_split


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(
    response: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ObservationDataset(response, labels, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _epoch(
    model: UWBFatModel,
    loader: DataLoader,
    frequency_hz: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: list[float] = []
    reconstructions: list[float] = []
    calibrations: list[float] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for response, caliper, _ in loader:
            response = response.to(device)
            caliper = caliper.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(response, frequency_hz)
            terms = physics_informed_loss(
                output,
                response,
                caliper,
                config.training.reconstruction_weight,
                config.training.calibration_weight,
                config.training.reconstruction_normalization,
            )
            if optimizer is not None:
                terms.total.backward()
                optimizer.step()
            totals.append(float(terms.total.detach()))
            reconstructions.append(float(terms.reconstruction.detach()))
            calibrations.append(float(terms.calibration.detach()))
    return {
        "total": float(np.mean(totals)),
        "reconstruction": float(np.mean(reconstructions)),
        "calibration": float(np.mean(calibrations)),
    }


def train_fold(
    dataset_path: str | Path,
    held_out: str,
    output_dir: str | Path,
    config: ProjectConfig,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Train one LOSO fold and evaluate the held-out participant by grid search."""
    _set_seed(config.training.seed)
    device = torch.device(device_name)
    archive = ObservationArchive(dataset_path)
    if archive.response.shape[2] != config.acquisition.inband_bins:
        raise ValueError("Archive frequency dimension does not match the configured model")
    split = full_loso_split(
        archive.participant_id,
        held_out,
        config.training.validation_fraction,
        config.training.seed,
    )

    normalizer = ComplexStandardizer().fit(archive.response[split.train])
    normalized = normalizer.transform(archive.response).astype(np.complex64)
    train_loader = _loader(
        normalized,
        archive.caliper_mm,
        split.train,
        config.training.batch_size,
        True,
    )
    validation_loader = _loader(
        normalized,
        archive.caliper_mm,
        split.validation,
        config.training.batch_size,
        False,
    )

    model = UWBFatModel(config.model, config.bounds).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    frequency_hz = torch.as_tensor(archive.frequencies_hz, dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_result = _epoch(model, train_loader, frequency_hz, config, device, optimizer)
        validation_result = _epoch(
            model, validation_loader, frequency_hz, config, device, optimizer=None
        )
        history.append({"epoch": epoch, "train": train_result, "validation": validation_result})
        if validation_result["total"] < best_validation:
            best_validation = validation_result["total"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    predictions = np.empty(len(split.test), dtype=np.float32)
    residuals = np.empty(len(split.test), dtype=np.float32)
    test_loader = _loader(
        normalized,
        archive.caliper_mm,
        split.test,
        config.training.batch_size,
        False,
    )
    cursor = 0
    with torch.no_grad():
        for response, _, _ in test_loader:
            result = grid_search(model, response.to(device), frequency_hz, config.bounds)
            size = response.shape[0]
            predictions[cursor : cursor + size] = result["caliper_mm"].cpu().numpy()
            residuals[cursor : cursor + size] = result["reconstruction_mse"].cpu().numpy()
            cursor += size

    references = archive.caliper_mm[split.test]
    window_metrics = regression_metrics(references, predictions)
    rec_reference, rec_prediction, _ = aggregate_by_recording(
        references, predictions, archive.recording_id[split.test]
    )
    recording_metrics = regression_metrics(rec_reference, rec_prediction)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in best_state.items()},
        "normalization_scale": normalizer.scale,
        "frequency_hz": archive.frequencies_hz,
        "held_out_participant": str(held_out),
        "config": asdict(config),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "participant_id",
                "recording_id",
                "site",
                "reference_mm",
                "prediction_mm",
                "reconstruction_mse",
            ]
        )
        for local, archive_index in enumerate(split.test):
            writer.writerow(
                [
                    archive.participant_id[archive_index],
                    archive.recording_id[archive_index],
                    archive.site[archive_index],
                    float(references[local]),
                    float(predictions[local]),
                    float(residuals[local]),
                ]
            )

    report: dict[str, object] = {
        "held_out_participant": str(held_out),
        "train_participants": sorted(set(archive.participant_id[split.train].astype(str))),
        "validation_participants": sorted(
            set(archive.participant_id[split.validation].astype(str))
        ),
        "test_participants": sorted(set(archive.participant_id[split.test].astype(str))),
        "trainable_parameters": trainable_parameter_count(model),
        "normalization_scale_from_outer_training_only": normalizer.scale,
        "best_validation_loss": best_validation,
        "epochs_completed": len(history),
        "window_level": window_metrics,
        "recording_level": recording_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return report
