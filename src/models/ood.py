from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.validation.schema_contract import ModelSchemaContract


@dataclass(frozen=True)
class OODAssessment:
    compatible: bool
    score: float
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    comparable_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_ood(values: dict[str, Any], contract: ModelSchemaContract) -> OODAssessment:
    blockers: list[str] = []
    warnings: list[str] = []
    checked = 0
    outside = 0
    for spec in contract.features:
        value = values.get(spec.name)
        if value in (None, ""):
            if spec.required:
                blockers.append(f"Required field '{spec.label}' is missing.")
            continue
        checked += 1
        if spec.dtype == "numeric":
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if spec.minimum is not None and spec.maximum is not None and not spec.minimum <= numeric <= spec.maximum:
                outside += 1
                span = max(spec.maximum - spec.minimum, abs(spec.median or 1.0), 1e-9)
                distance = spec.minimum - numeric if numeric < spec.minimum else numeric - spec.maximum
                message = f"{spec.label} is outside training coverage [{spec.minimum:g}, {spec.maximum:g}]."
                (blockers if distance / span > 0.20 else warnings).append(message)
        elif spec.vocabulary and str(value) not in set(spec.vocabulary):
            outside += 1
            message = f"{spec.label} value '{value}' was not observed during training."
            if spec.normalized_role == "property_type":
                blockers.append(message + " No compatible property-type model can be assumed.")
            else:
                warnings.append(message)
    score = outside / max(checked, 1)
    if score > 0.35:
        blockers.append("Too many inputs fall outside this model's training coverage.")
    return OODAssessment(not blockers, round(score, 4), tuple(blockers), tuple(warnings), round(max(0.0, 1.0 - score), 4))
