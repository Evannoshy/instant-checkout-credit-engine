"""Late-fusion prediction handoff (fusion-stacker-v1): one loan's base scores in, one PD out.

``load_predictor(variant)`` rebuilds a fitted variant from its JSON artifact
(``data/fusion/stacker_v1/<variant>.json``) and the approved calibrated
tabular-only mapping (the ``tabular_calibration_v1`` Platt parameters for the
balanced v1 LR, pinned by SHA-256 in the stacker config).

``predict(predictor, scores, text_usable=...)`` returns::

    {"p_default", "source", "fallback", "text_usable", "tracks_used", "model_version"}

- ``text_usable=True``: every track of the variant must have a finite score in
  [0, 1]. The result is the calibrated stacker output, ``fallback=False``.
- ``text_usable=False``: the result is the approved calibrated tabular-only
  score, ``source="tabular_fallback"`` and ``fallback=True``. A text score
  passed alongside is refused: the caller has said there is no usable text.

A missing, non-finite or out-of-range score raises ValueError. No score is
ever filled, and no text-model probability is invented.

Inputs are base-model probabilities, not borrower fields: the saved tabular
pipeline (Brandon) and the text scorer (Vincent's ``score_borrower_text``)
produce them upstream.

Known limitations:
- Whether text is usable is the caller's decision. Evan and Ariel have not yet
  confirmed the definition (empty description vs empty cleaned
  title-purpose-description payload), so this module does not decide it.
- The committed variant is ``tabular_only``; text variants need the v2 NLP exports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, logit

from src.fusion import inputs
from src.fusion.stacker import DEFAULT_ARTIFACT_DIR
from src.tabular import provenance
from src.tabular.preprocess import REPO_ROOT

TABULAR_TRACK = "tabular"
REFERENCE_CANDIDATE = "logistic_regression"
SOURCE_STACKER = "fusion_stacker"
SOURCE_FALLBACK = "tabular_fallback"

# Every ValueError message raised by this module, as str.format templates.
ERROR_MESSAGES: dict[str, str] = {
    "text_flag": "text_usable must be True or False; got {value!r}",
    "unknown_tracks": "Scores for unknown tracks {unknown}; this variant uses {known}",
    "missing_score": "No {track} score; a missing score is never filled",
    "bad_score": "{track} score must be a finite probability in [0, 1]; got {value!r}",
    "text_score_without_text": (
        "Text marked unusable but {tracks} scores were given; no text-model probability is used then"
    ),
    "artifact_mismatch": "Artifact {path} is for {actual!r}, not {expected!r}",
}


@dataclass(frozen=True)
class PlattMapping:
    """sigmoid(slope * logit(clip(p)) + intercept), as fitted by ``calibrate.fit_platt``."""

    slope: float
    intercept: float
    clip: float

    def apply(self, p: float) -> float:
        clipped = min(max(p, self.clip), 1 - self.clip)
        return float(expit(self.slope * logit(clipped) + self.intercept))


@dataclass(frozen=True)
class FusionPredictor:
    stacker_version: str
    variant: str
    tracks: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibrator: PlattMapping
    fallback: PlattMapping
    fallback_version: str

    def stacker_probability(self, values: list[float]) -> float:
        """The raw stacker output: logistic regression on the probabilities, as fitted."""
        return float(expit(np.dot(self.coefficients, values) + self.intercept))


def _reference_platt(config: dict[str, Any], root: Path) -> PlattMapping:
    """The approved tabular_calibration_v1 Platt mapping for the balanced v1 LR."""
    reference = config["reference"]
    path = root / reference["calibration_metrics_path"]
    provenance.verify_file_checksum(path, reference["calibration_metrics_sha256"], allow_crlf=True)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    params = metrics["candidates"][REFERENCE_CANDIDATE]["calibrators"]["platt"]
    return PlattMapping(params["slope"], params["intercept"], metrics["settings"]["probability_clip"])


def load_predictor(
    variant: str = "tabular_only",
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    config_path: str | Path = inputs.DEFAULT_STACKER_CONFIG,
    root: Path = REPO_ROOT,
) -> FusionPredictor:
    """Rebuild one fitted variant and the approved tabular fallback from committed files."""
    config = inputs.load_stacker_config(config_path)
    path = Path(artifact_dir) / f"{variant}.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["variant"] != variant or artifact["stacker_version"] != config["stacker_version"]:
        raise ValueError(ERROR_MESSAGES["artifact_mismatch"].format(
            path=path, actual=(artifact["variant"], artifact["stacker_version"]),
            expected=(variant, config["stacker_version"]),
        ))
    calibrator = artifact["calibrator"]
    return FusionPredictor(
        stacker_version=artifact["stacker_version"],
        variant=variant,
        tracks=tuple(artifact["tracks"]),
        coefficients=tuple(artifact["stacker"]["coefficients"]),
        intercept=artifact["stacker"]["intercept"],
        calibrator=PlattMapping(calibrator["slope"], calibrator["intercept"], calibrator["probability_clip"]),
        fallback=_reference_platt(config, root),
        fallback_version=config["reference"]["calibration_version"],
    )


def _checked_score(track: str, value: Any) -> float:
    if value is None:
        raise ValueError(ERROR_MESSAGES["missing_score"].format(track=track))
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(ERROR_MESSAGES["bad_score"].format(track=track, value=value))
    return float(value)


def predict(predictor: FusionPredictor, scores: Mapping[str, float | None], *, text_usable: bool) -> dict[str, Any]:
    """One loan's PD from its base-model probabilities, or the tabular fallback when text is unusable."""
    if not isinstance(text_usable, bool):
        raise ValueError(ERROR_MESSAGES["text_flag"].format(value=text_usable))
    known = {TABULAR_TRACK, *predictor.tracks}
    if unknown := sorted(set(scores) - known):
        raise ValueError(ERROR_MESSAGES["unknown_tracks"].format(unknown=unknown, known=sorted(known)))
    tabular = _checked_score(TABULAR_TRACK, scores.get(TABULAR_TRACK))

    if not text_usable:
        text_given = sorted(track for track, value in scores.items() if track != TABULAR_TRACK and value is not None)
        if text_given:
            raise ValueError(ERROR_MESSAGES["text_score_without_text"].format(tracks=text_given))
        return {
            "p_default": predictor.fallback.apply(tabular),
            "source": SOURCE_FALLBACK,
            "fallback": True,
            "text_usable": False,
            "tracks_used": [TABULAR_TRACK],
            "model_version": predictor.fallback_version,
        }

    values = [_checked_score(track, scores.get(track)) for track in predictor.tracks]
    return {
        "p_default": predictor.calibrator.apply(predictor.stacker_probability(values)),
        "source": SOURCE_STACKER,
        "fallback": False,
        "text_usable": True,
        "tracks_used": list(predictor.tracks),
        "model_version": f"{predictor.stacker_version}/{predictor.variant}",
    }
