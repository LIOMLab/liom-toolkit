"""Registration subpackage: ANTs-based volume registration and template building."""

from __future__ import annotations

from .register import (
    align_annotations_to_volume,
    align_brain_region_to_atlas,
    align_volume_to_allen,
)
from .templating import (
    build_template_for_resolution,
)

__all__ = [
    "align_annotations_to_volume",
    "align_brain_region_to_atlas",
    "align_volume_to_allen",
    "build_template_for_resolution",
]
