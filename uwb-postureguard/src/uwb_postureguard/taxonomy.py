"""Canonical 19-class posture taxonomy."""

from __future__ import annotations

from enum import IntEnum
from numbers import Integral, Real


class Posture(IntEnum):
    IDLE = 0
    UPRIGHT = 1
    LEANING_FORWARD = 2
    LEANING_BACK = 3
    LATERAL_LEAN_LEFT = 4
    CROSS_LEG_LEFT = 5
    LATERAL_LEAN_RIGHT = 6
    CROSS_LEG_RIGHT = 7
    HUNCHED = 8
    TENSE = 9
    LIE_ON_TABLE = 10
    ROTATE_HEAD = 11
    VERTICAL_LEG_SHAKING_LEFT = 12
    VERTICAL_LEG_SHAKING_RIGHT = 13
    HORIZONTAL_LEG_SHAKING = 14
    TAP_FINGER = 15
    STRETCH = 16
    STAND = 17
    WALK = 18


POSTURES: dict[int, str] = {
    Posture.IDLE: "idle",
    Posture.UPRIGHT: "upright",
    Posture.LEANING_FORWARD: "leaning forward",
    Posture.LEANING_BACK: "leaning back",
    Posture.LATERAL_LEAN_LEFT: "lateral lean left",
    Posture.CROSS_LEG_LEFT: "cross leg left",
    Posture.LATERAL_LEAN_RIGHT: "lateral lean right",
    Posture.CROSS_LEG_RIGHT: "cross leg right",
    Posture.HUNCHED: "hunched",
    Posture.TENSE: "tense",
    Posture.LIE_ON_TABLE: "lie on table",
    Posture.ROTATE_HEAD: "rotate head",
    Posture.VERTICAL_LEG_SHAKING_LEFT: "vertical leg shaking left",
    Posture.VERTICAL_LEG_SHAKING_RIGHT: "vertical leg shaking right",
    Posture.HORIZONTAL_LEG_SHAKING: "horizontal leg shaking",
    Posture.TAP_FINGER: "tap finger",
    Posture.STRETCH: "stretch",
    Posture.STAND: "stand",
    Posture.WALK: "walk",
}

STATIC_POSTURES = frozenset(range(11))
DYNAMIC_POSTURES = frozenset(range(11, 19))


def _normal_form(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


_ALIASES = {
    "hunch": Posture.HUNCHED,
    "lateral lean (left)": Posture.LATERAL_LEAN_LEFT,
    "lateral lean (right)": Posture.LATERAL_LEAN_RIGHT,
    "cross leg (left)": Posture.CROSS_LEG_LEFT,
    "cross leg (right)": Posture.CROSS_LEG_RIGHT,
    "rotate the neck": Posture.ROTATE_HEAD,
    "rotate neck": Posture.ROTATE_HEAD,
    "vert leg shake left": Posture.VERTICAL_LEG_SHAKING_LEFT,
    "vert leg shake right": Posture.VERTICAL_LEG_SHAKING_RIGHT,
    "horiz leg shaking": Posture.HORIZONTAL_LEG_SHAKING,
}

_NAME_TO_ID = {_normal_form(name): int(posture_id) for posture_id, name in POSTURES.items()}
_NAME_TO_ID.update({_normal_form(name): int(posture_id) for name, posture_id in _ALIASES.items()})


def normalize_posture(value: object) -> int:
    """Return a canonical class id from an integer id or a known label string."""

    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid posture labels")
    if isinstance(value, Integral) or isinstance(value, Real) and float(value).is_integer():
        label = int(value)
    else:
        text = _normal_form(str(value))
        if text.isdigit():
            label = int(text)
        elif text in _NAME_TO_ID:
            label = _NAME_TO_ID[text]
        else:
            raise ValueError(f"Unknown posture label: {value!r}")
    if label not in POSTURES:
        raise ValueError(f"Posture id must be in [0, 18], got {label}")
    return label
