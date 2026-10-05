# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression: lite vs hardened threshold sections must stay independently calibrated."""

from __future__ import annotations

from asha.core.ml.calibration import get_bands, load_thresholds


def test_mission_domain_lite_and_hardened_thresholds_differ() -> None:
    """Accidental re-merge of shared cutoffs must fail loudly."""
    cfg = load_thresholds()
    hardened = get_bands("mission_domain", thresholds=cfg)
    lite = get_bands("mission_domain_lite", thresholds=cfg)
    assert "mission_domain" in cfg
    assert "mission_domain_lite" in cfg
    assert hardened.safe_max != lite.safe_max, (
        "mission_domain and mission_domain_lite share the same safe_max; "
        "recalibrate each path independently (do not copy one value into both)"
    )


def test_intent_and_optimizer_split_sections_exist() -> None:
    cfg = load_thresholds()
    for key in (
        "intent_extraction",
        "intent_extraction_lite",
        "optimizer_similarity",
        "optimizer_similarity_lite",
    ):
        assert key in cfg, f"missing threshold section: {key}"
        bands = get_bands(key, thresholds=cfg)
        assert bands.safe_max > 0.0
