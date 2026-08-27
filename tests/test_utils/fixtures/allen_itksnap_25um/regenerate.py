"""One-time fixture regeneration for the 25µm ITK-SNAP label regression test.

This script is NOT collected by pytest (no ``test_`` prefix, no pytest
functions). It is a gated one-time dev script that generates the committed
allensdk snapshot fixture (``labels.parquet`` + ``annotation_25um.npz``) used
by ``tests/test_utils/test_allen_sdk.py::test_export_itksnap_labels_25um_matches_allensdk_fixture``.

Run manually on Python 3.11 with allensdk 2.16.x installed. allensdk does NOT
install on Python 3.12+ — its numpy 1.23.5 pin relies on ``numpy.distutils``,
which was removed in Python 3.12. Use a throwaway 3.11 venv:

    python3.11 -m venv /tmp/allensdk-test
    /tmp/allensdk-test/bin/pip install allensdk==2.16.2 nrrd pandas pyarrow
    /tmp/allensdk-test/bin/python tests/test_utils/fixtures/allen_itksnap_25um/regenerate.py

This downloads the 25µm annotation NRRD + structure-tree JSON via allensdk's
``ReferenceSpaceCache`` (the exact code path being replaced), calls
``rs.export_itksnap_labels()``, and saves the resulting DataFrame + volume as
the permanent regression oracle. After committing the fixture, allensdk is
never needed again — the regression test reads the committed fixture and
compares it against the rewritten ``liom_toolkit.utils.allen_sdk`` output.

The script also saves the raw ``structure_tree.json`` and ``annotation_25.nrrd``
to the fixture directory so the regression test can replay the rewrite without
network (it points ``construct_reference_space`` at the fixture dir, which has
the cached files). The cached ``structure_tree.json`` is allensdk's
post-``clean_structures`` flat list (each node has ``rgb_triplet`` as ints,
not ``color_hex_triplet``) — the rewrite's read path handles this format
directly.
"""

from __future__ import annotations

import os

import numpy as np

# One-time, gated — allensdk is only imported here, never in the package.
from allensdk.core.reference_space import ReferenceSpace
from allensdk.core.reference_space_cache import ReferenceSpaceCache

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    resolution = 25

    # Use the fixture directory as the data_dir so the cached NRRD + JSON land
    # alongside the parquet + npz fixtures.
    rsc = ReferenceSpaceCache(
        resolution=resolution,
        reference_space_key="annotation/ccf_2017",
        manifest=os.path.join(FIXTURE_DIR, "manifest.json"),
    )

    # Download annotation volume (writes annotation_25.nrrd to the fixture dir)
    annotation, _meta = rsc.get_annotation_volume(
        file_name=os.path.join(FIXTURE_DIR, "annotation_25.nrrd")
    )

    # Download structure tree (writes structure_tree.json to the fixture dir)
    structure_tree = rsc.get_structure_tree(
        file_name=os.path.join(FIXTURE_DIR, "structure_tree.json")
    )

    # Build the reference space and export ITK-SNAP labels
    rs = ReferenceSpace(
        resolution=resolution,
        annotation=annotation,
        structure_tree=structure_tree,
    )
    vol, df = rs.export_itksnap_labels()

    # Save the fixture artifacts
    df.to_parquet(os.path.join(FIXTURE_DIR, "labels.parquet"))
    np.savez_compressed(os.path.join(FIXTURE_DIR, "annotation_25um.npz"), arr=vol)

    print(f"Fixture generated in {FIXTURE_DIR}:")
    print(f"  labels.parquet: {len(df)} rows, {df.shape[1]} columns")
    print(f"  annotation_25um.npz: shape={vol.shape}, dtype={vol.dtype}")
    print("  annotation_25.nrrd: cached for the regression test")
    print("  structure_tree.json: cached for the regression test")


if __name__ == "__main__":
    main()
