"""Strict parsers for ranging-frame and CIR binary payloads used by the legacy collector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def extract_q(
    byte_array: bytes, integer_bits: int, fractional_bits: int, *, signed: bool = False
) -> float:
    expected_bits = integer_bits + fractional_bits
    if len(byte_array) * 8 != expected_bits:
        raise ValueError(
            f"Q-format payload has {len(byte_array) * 8} bits; expected {expected_bits}"
        )
    raw = int.from_bytes(byte_array, byteorder="little", signed=signed)
    return raw / float(1 << fractional_bits)


def twos_complement(value: int, bits: int) -> int:
    if bits < 1 or value < 0 or value >= 1 << bits:
        raise ValueError("value must fit in the requested unsigned bit width")
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


@dataclass(frozen=True, slots=True)
class CIRSamples:
    real: np.ndarray
    imaginary: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray

    @property
    def complex(self) -> np.ndarray:
        return self.real.astype(float) + 1j * self.imaginary.astype(float)


def parse_cir_samples(payload: bytes) -> CIRSamples:
    if len(payload) % 4:
        raise ValueError("Each CIR sample must contain two little-endian int16 values")
    pairs = np.frombuffer(payload, dtype="<i2").reshape(-1, 2).astype(np.int32, copy=True)
    real = pairs[:, 0]
    imaginary = pairs[:, 1]
    complex_values = real.astype(float) + 1j * imaginary.astype(float)
    return CIRSamples(
        real=real,
        imaginary=imaginary,
        magnitude=np.abs(complex_values),
        phase=np.angle(complex_values),
    )


def _rx_timestamp(payload: bytes) -> float:
    if len(payload) != 6:
        raise ValueError("RX timestamp must contain exactly six bytes")
    integer = int.from_bytes(payload[:4], byteorder="little")
    fraction = int.from_bytes(payload[4:], byteorder="little") / float(1 << 9)
    return integer + fraction


@dataclass(frozen=True, slots=True)
class RangingMeasurement:
    slot_index: int
    rx_index: int
    decode_status: int
    nlos: int
    first_path_index: float
    main_path_index: float
    snr_main_path: int
    snr_first_path: int
    snr_total: float
    rssi: float
    cir_main_power: int
    cir_first_path_power: int
    noise_variance: int
    cfo: int
    aoa_phase: float
    cir: CIRSamples
    rx_timestamp: float

    @classmethod
    def from_bytes(cls, payload: bytes) -> RangingMeasurement:
        if len(payload) < 37:
            raise ValueError("Ranging measurement is too short")
        mapping = payload[0]
        cir_payload = payload[27:-6]
        return cls(
            slot_index=mapping & 0x3F,
            rx_index=(mapping >> 7) & 0x01,
            decode_status=payload[1],
            nlos=payload[2],
            first_path_index=extract_q(payload[3:5], 10, 6),
            main_path_index=extract_q(payload[5:7], 10, 6),
            snr_main_path=payload[7],
            snr_first_path=payload[8],
            snr_total=extract_q(payload[9:11], 8, 8),
            rssi=extract_q(payload[11:13], 8, 8),
            cir_main_power=int.from_bytes(payload[13:17], "little"),
            cir_first_path_power=int.from_bytes(payload[17:21], "little"),
            noise_variance=int.from_bytes(payload[21:23], "little"),
            cfo=int.from_bytes(payload[23:25], "little", signed=True),
            aoa_phase=extract_q(payload[25:27], 9, 7),
            cir=parse_cir_samples(cir_payload),
            rx_timestamp=_rx_timestamp(payload[-6:]),
        )

    def to_features(self, role: str, measurement_index: int) -> dict[str, float | int]:
        prefix = f"{role}_"
        output: dict[str, float | int] = {
            f"{prefix}nlos_{measurement_index}": self.nlos,
            f"{prefix}first_path_idx_{measurement_index}": self.first_path_index,
            f"{prefix}main_path_idx_{measurement_index}": self.main_path_index,
            f"{prefix}snr_main_path_{measurement_index}": self.snr_main_path,
            f"{prefix}snr_1st_path_{measurement_index}": self.snr_first_path,
            f"{prefix}snr_total_{measurement_index}": self.snr_total,
            f"{prefix}rssi_{measurement_index}": self.rssi,
            f"{prefix}cir_main_pwr_{measurement_index}": self.cir_main_power,
            f"{prefix}cir_1st_path_pwr_{measurement_index}": self.cir_first_path_power,
            f"{prefix}noise_variance_{measurement_index}": self.noise_variance,
            f"{prefix}cfo_{measurement_index}": self.cfo,
            f"{prefix}aoa_phase_{measurement_index}": self.aoa_phase,
        }
        for sample_index, (real, imaginary, magnitude, phase) in enumerate(
            zip(self.cir.real, self.cir.imaginary, self.cir.magnitude, self.cir.phase)
        ):
            suffix = f"{measurement_index}_{sample_index}"
            output[f"{prefix}cir_re_{suffix}"] = int(real)
            output[f"{prefix}cir_im_{suffix}"] = int(imaginary)
            output[f"{prefix}cir_mag_{suffix}"] = float(magnitude)
            output[f"{prefix}cir_ang_{suffix}"] = float(phase)
        return output


@dataclass(frozen=True, slots=True)
class RangingFrameLog:
    session_handle: int
    measurement_size: int
    measurements: tuple[RangingMeasurement, ...]

    @classmethod
    def from_bytes(cls, payload: bytes) -> RangingFrameLog:
        if len(payload) < 6:
            raise ValueError("Ranging frame log is too short")
        session_handle = int.from_bytes(payload[:4], "little")
        measurement_count = payload[4]
        measurement_size = payload[5]
        expected = 6 + measurement_count * measurement_size
        if len(payload) != expected:
            raise ValueError(f"Ranging frame log has {len(payload)} bytes; expected {expected}")
        measurements = tuple(
            RangingMeasurement.from_bytes(
                payload[6 + index * measurement_size : 6 + (index + 1) * measurement_size]
            )
            for index in range(measurement_count)
        )
        return cls(session_handle, measurement_size, measurements)

    @classmethod
    def from_file(cls, path: str | Path) -> RangingFrameLog:
        return cls.from_bytes(Path(path).read_bytes())

    def to_features(self, role: str) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        for index, measurement in enumerate(self.measurements, start=1):
            output.update(measurement.to_features(role, index))
        return output


@dataclass(frozen=True, slots=True)
class CIRFrame:
    session_handle: int
    slot_handle: int
    rx_antenna_id: int
    samples: CIRSamples

    @classmethod
    def from_bytes(cls, payload: bytes) -> CIRFrame:
        if len(payload) < 8:
            raise ValueError("CIR frame is too short")
        sample_count = int.from_bytes(payload[6:8], "little")
        expected = 8 + sample_count * 4
        if len(payload) != expected:
            raise ValueError(f"CIR frame has {len(payload)} bytes; expected {expected}")
        return cls(
            session_handle=int.from_bytes(payload[:4], "little"),
            slot_handle=payload[4],
            rx_antenna_id=payload[5],
            samples=parse_cir_samples(payload[8:]),
        )


@dataclass(frozen=True, slots=True)
class CIRLog:
    frames: tuple[CIRFrame, ...]

    @classmethod
    def from_bytes(cls, payload: bytes) -> CIRLog:
        frames: list[CIRFrame] = []
        offset = 0
        while offset < len(payload):
            if len(payload) - offset < 8:
                raise ValueError("Trailing bytes do not form a complete CIR frame header")
            sample_count = int.from_bytes(payload[offset + 6 : offset + 8], "little")
            frame_size = 8 + sample_count * 4
            end = offset + frame_size
            if end > len(payload):
                raise ValueError("CIR frame extends beyond the available payload")
            frames.append(CIRFrame.from_bytes(payload[offset:end]))
            offset = end
        return cls(tuple(frames))

    @classmethod
    def from_file(cls, path: str | Path) -> CIRLog:
        return cls.from_bytes(Path(path).read_bytes())

    def to_features(self, role: str) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        for frame in self.frames:
            for index, (real, imaginary, magnitude, phase) in enumerate(
                zip(
                    frame.samples.real,
                    frame.samples.imaginary,
                    frame.samples.magnitude,
                    frame.samples.phase,
                ),
                start=1,
            ):
                stem = f"{role}_cir_{frame.rx_antenna_id}"
                output[f"{stem}_re_{index}"] = int(real)
                output[f"{stem}_im_{index}"] = int(imaginary)
                output[f"{stem}_mag_{index}"] = float(magnitude)
                output[f"{stem}_ang_{index}"] = float(phase)
        return output
