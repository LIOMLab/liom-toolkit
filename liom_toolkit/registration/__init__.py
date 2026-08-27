from __future__ import annotations

from .register import (
    align_annotations_to_volume,
    align_brain_region_to_atlas,
    align_volume_to_allen,
    deformably_register_volume,
    get_transformations_for_atlas,
    rigidly_register_volume,
)
from .templating import (
    build_template,
    build_template_for_resolution,
    create_template,
    load_volume_for_registration,
    pre_register_brain,
    update_brain_name_list,
)

__all__ = [
    # register
    "align_annotations_to_volume",
    "align_brain_region_to_atlas",
    "align_volume_to_allen",
    "deformably_register_volume",
    "get_transformations_for_atlas",
    "rigidly_register_volume",
    # templating
    "build_template",
    "build_template_for_resolution",
    "create_template",
    "load_volume_for_registration",
    "pre_register_brain",
    "update_brain_name_list",
]
