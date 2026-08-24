"""Jackson-Pollock three-site density equations and Siri conversion."""

from __future__ import annotations

MALE_SITES = ("chest", "abdomen", "thigh")
FEMALE_SITES = ("triceps", "suprailiac", "thigh")


def jackson_pollock_density(sex: str, age_years: float, site_values_mm: dict[str, float]) -> float:
    sex = sex.strip().lower()
    if sex in {"m", "male"}:
        sites = MALE_SITES
        total = sum(float(site_values_mm[site]) for site in sites)
        return 1.10938 - 0.0008267 * total + 0.0000016 * total**2 - 0.0002574 * age_years
    if sex in {"f", "female"}:
        sites = FEMALE_SITES
        total = sum(float(site_values_mm[site]) for site in sites)
        return 1.0994921 - 0.0009929 * total + 0.0000023 * total**2 - 0.0001392 * age_years
    raise ValueError("sex must be 'male' or 'female'")


def siri_percent(body_density: float) -> float:
    if body_density <= 0:
        raise ValueError("body_density must be positive")
    return (4.95 / body_density - 4.50) * 100.0


def body_fat_percent(sex: str, age_years: float, site_values_mm: dict[str, float]) -> float:
    return siri_percent(jackson_pollock_density(sex, age_years, site_values_mm))
