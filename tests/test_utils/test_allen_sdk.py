"""Tests for ``liom_toolkit/utils/allen_sdk.py``.

Phase 4 migration: ``allen_sdk.py`` is rewritten to remove the ``allensdk``
dependency and replace it with direct HTTP download of Allen Institute CCFv3
NRRD volumes + structure-tree JSON. These tests cover the pure-logic helpers
that reproduce the allensdk ``export_itksnap_labels`` semantics (D-04
byte-exactness), plus mock-network tests for the download path and an
import-smoke test that asserts the module loads without ``allensdk`` installed.

The pure-logic tests use small synthetic structure-tree dicts and small numpy
annotation arrays — no network, no allensdk, no ants. The mock-network tests
patch ``requests.get`` so no real HTTP call is made. The 25um regression test
(Task 3) reads a committed allensdk snapshot fixture and asserts byte-exact
reproduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Task 1 — pure-logic helpers
# ---------------------------------------------------------------------------


def _sample_structure_tree() -> list[dict]:
    """A 3-level nested structure-tree ``msg`` payload (root + child + grandchild).

    Mirrors the Allen ``structure_graph_download/1.json`` shape: each node has
    ``id``, ``acronym``, ``name``, ``color_hex_triplet``, ``structure_id_path``
    and a ``children`` array of descendant nodes.
    """
    return [
        {
            "id": 997,
            "acronym": "root",
            "name": "root",
            "color_hex_triplet": "019393",
            "structure_id_path": "997",
            "children": [
                {
                    "id": 8,
                    "acronym": "grey",
                    "name": "Basic cell groups and regions",
                    "color_hex_triplet": "FF0000",
                    "structure_id_path": "997/8",
                    "children": [],
                },
            ],
        }
    ]


class TestHexToRgb:
    """``_hex_to_rgb`` mirrors allensdk v2.16.2 ``hex_to_rgb`` semantics."""

    def test_six_char_string(self):
        from liom_toolkit.utils.allen_sdk import _hex_to_rgb

        # "019393" -> 0x01=1, 0x93=147, 0x93=147
        assert _hex_to_rgb("019393") == [1, 147, 147]

    def test_leading_hash_stripped(self):
        from liom_toolkit.utils.allen_sdk import _hex_to_rgb

        assert _hex_to_rgb("#019393") == [1, 147, 147]

    def test_short_string_padded_to_six(self):
        from liom_toolkit.utils.allen_sdk import _hex_to_rgb

        # "0" -> "000000" -> [0, 0, 0]
        assert _hex_to_rgb("0") == [0, 0, 0]

    def test_empty_string_padded_to_six(self):
        from liom_toolkit.utils.allen_sdk import _hex_to_rgb

        assert _hex_to_rgb("") == [0, 0, 0]

    def test_full_white(self):
        from liom_toolkit.utils.allen_sdk import _hex_to_rgb

        assert _hex_to_rgb("FFFFFF") == [255, 255, 255]


class TestFlattenStructureTree:
    """``_flatten_structure_tree`` walks the nested ``children`` hierarchy depth-first."""

    def test_three_level_nested_returns_flat_list_in_msg_order(self):
        from liom_toolkit.utils.allen_sdk import _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        # Root first, then child (depth-first, JSON msg order)
        assert len(flat) == 2
        assert flat[0]["id"] == 997
        assert flat[0]["acronym"] == "root"
        assert flat[1]["id"] == 8
        assert flat[1]["acronym"] == "grey"

    def test_attaches_rgb_triplet(self):
        from liom_toolkit.utils.allen_sdk import _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        assert flat[0]["rgb_triplet"] == [1, 147, 147]
        assert flat[1]["rgb_triplet"] == [255, 0, 0]

    def test_preserves_original_node_fields(self):
        from liom_toolkit.utils.allen_sdk import _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        # Original fields are preserved alongside the new rgb_triplet
        assert flat[0]["color_hex_triplet"] == "019393"
        assert flat[0]["structure_id_path"] == "997"
        assert flat[1]["structure_id_path"] == "997/8"

    def test_empty_children_returns_root_only(self):
        from liom_toolkit.utils.allen_sdk import _flatten_structure_tree

        msg = [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "019393",
                "structure_id_path": "997",
                "children": [],
            }
        ]
        flat = _flatten_structure_tree(msg)
        assert len(flat) == 1
        assert flat[0]["id"] == 997

    def test_deeper_nesting_preserves_depth_first_order(self):
        from liom_toolkit.utils.allen_sdk import _flatten_structure_tree

        msg = [
            {
                "id": 1,
                "acronym": "a",
                "name": "a",
                "color_hex_triplet": "0",
                "structure_id_path": "1",
                "children": [
                    {
                        "id": 2,
                        "acronym": "b",
                        "name": "b",
                        "color_hex_triplet": "0",
                        "structure_id_path": "1/2",
                        "children": [
                            {
                                "id": 3,
                                "acronym": "c",
                                "name": "c",
                                "color_hex_triplet": "0",
                                "structure_id_path": "1/2/3",
                                "children": [],
                            }
                        ],
                    },
                    {
                        "id": 4,
                        "acronym": "d",
                        "name": "d",
                        "color_hex_triplet": "0",
                        "structure_id_path": "1/4",
                        "children": [],
                    },
                ],
            }
        ]
        flat = _flatten_structure_tree(msg)
        # Depth-first: a, b, c, d
        assert [n["id"] for n in flat] == [1, 2, 3, 4]


class TestBuildStructureMetadata:
    """``_build_structure_metadata`` builds the 8-column ITK-SNAP label DataFrame."""

    def test_two_structures_eight_columns_in_order(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        df = _build_structure_metadata(flat)
        assert list(df.columns) == ["IDX", "-R-", "-G-", "-B-", "-A-", "VIS", "MSH", "LABEL"]

    def test_column_values_match_structure_fields(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        df = _build_structure_metadata(flat)
        # Row 0: root (id=997, rgb=[1,147,147], acronym="root")
        assert df.loc[0, "IDX"] == 997
        assert df.loc[0, "-R-"] == 1
        assert df.loc[0, "-G-"] == 147
        assert df.loc[0, "-B-"] == 147
        assert df.loc[0, "-A-"] == 1.0
        assert df.loc[0, "VIS"] == 1
        assert df.loc[0, "MSH"] == 1
        assert df.loc[0, "LABEL"] == "root"
        # Row 1: grey (id=8, rgb=[255,0,0], acronym="grey")
        assert df.loc[1, "IDX"] == 8
        assert df.loc[1, "-R-"] == 255
        assert df.loc[1, "-G-"] == 0
        assert df.loc[1, "-B-"] == 0
        assert df.loc[1, "LABEL"] == "grey"

    def test_dtypes(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        df = _build_structure_metadata(flat)
        assert str(df["IDX"].dtype) == "int64"
        assert str(df["-A-"].dtype) == "float64"
        assert str(df["VIS"].dtype) == "int64"
        assert str(df["MSH"].dtype) == "int64"
        # LABEL is a string column — pandas <3.0 infers `object`, pandas 3.0+
        # infers `StringDtype`. The allensdk fixture (generated on 3.12 with
        # older pandas) will have `object`; the byte-exact regression test
        # (Task 3) is the final arbiter. Accept either here.
        assert df["LABEL"].dtype == object or "str" in str(df["LABEL"].dtype).lower()

    def test_row_order_matches_flatten_order(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree

        flat = _flatten_structure_tree(_sample_structure_tree())
        df = _build_structure_metadata(flat)
        # IDX order = flatten order = JSON msg order
        assert list(df["IDX"].values) == [997, 8]
        assert list(df["LABEL"].values) == ["root", "grey"]


class TestRemapToIdType:
    """``_remap_to_id_type`` mirrors allensdk ``ReferenceSpace.export_itksnap_labels`` remap."""

    def test_no_remap_when_all_idx_within_range(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree, _remap_to_id_type

        flat = _flatten_structure_tree(_sample_structure_tree())
        df = _build_structure_metadata(flat)
        # Annotation volume with values that match IDX (all <= 65535)
        annotation = np.zeros((4, 4, 4), dtype=np.uint32)
        annotation[0, 0, 0] = 997
        annotation[1, 1, 1] = 8
        new_vol, new_df = _remap_to_id_type(annotation, df, id_type=np.uint16)
        # No remap: returned unchanged
        np.testing.assert_array_equal(new_vol, annotation)
        pd.testing.assert_frame_equal(new_df, df)

    def test_remap_when_idx_exceeds_uint16_max(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree, _remap_to_id_type

        # Build a structure tree with one id > 65535
        msg = [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "019393",
                "structure_id_path": "997",
                "children": [
                    {
                        "id": 100000,  # > 65535 -> triggers remap
                        "acronym": "zzz",
                        "name": "zzz",
                        "color_hex_triplet": "FF0000",
                        "structure_id_path": "997/100000",
                        "children": [],
                    },
                    {
                        "id": 8,
                        "acronym": "aaa",
                        "name": "aaa",
                        "color_hex_triplet": "00FF00",
                        "structure_id_path": "997/8",
                        "children": [],
                    },
                ],
            }
        ]
        flat = _flatten_structure_tree(msg)
        df = _build_structure_metadata(flat)
        annotation = np.zeros((4, 4, 4), dtype=np.uint32)
        annotation[0, 0, 0] = 100000
        annotation[1, 1, 1] = 8
        annotation[2, 2, 2] = 997
        new_vol, new_df = _remap_to_id_type(annotation, df, id_type=np.uint16)

        # Remap: sorted by LABEL -> "aaa"(id=8)=1, "root"(id=997)=2, "zzz"(id=100000)=3
        # New IDX is sequential 1..N
        assert list(new_df["LABEL"].values) == ["aaa", "root", "zzz"]
        assert list(new_df["IDX"].values) == [1, 2, 3]
        # Volume voxels remapped to match
        assert new_vol[1, 1, 1] == 1  # was 8 -> "aaa" -> 1
        assert new_vol[2, 2, 2] == 2  # was 997 -> "root" -> 2
        assert new_vol[0, 0, 0] == 3  # was 100000 -> "zzz" -> 3
        # Output volume dtype is id_type (uint16)
        assert new_vol.dtype == np.uint16

    def test_volume_idx_consistency_after_remap(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree, _remap_to_id_type

        msg = [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "0",
                "structure_id_path": "997",
                "children": [
                    {
                        "id": 70000,
                        "acronym": "leaf",
                        "name": "leaf",
                        "color_hex_triplet": "0",
                        "structure_id_path": "997/70000",
                        "children": [],
                    }
                ],
            }
        ]
        flat = _flatten_structure_tree(msg)
        df = _build_structure_metadata(flat)
        annotation = np.zeros((4, 4, 4), dtype=np.uint32)
        annotation[0, 0, 0] = 70000
        annotation[1, 1, 1] = 997
        new_vol, new_df = _remap_to_id_type(annotation, df, id_type=np.uint16)
        # Every unique non-zero value in new_vol equals exactly one IDX in the df
        unique_vol_values = set(np.unique(new_vol)) - {0}
        unique_idx_values = set(new_df["IDX"].values)
        assert unique_vol_values == unique_idx_values

    def test_remap_resets_index(self):
        from liom_toolkit.utils.allen_sdk import _build_structure_metadata, _flatten_structure_tree, _remap_to_id_type

        msg = [
            {
                "id": 70000,
                "acronym": "z",
                "name": "z",
                "color_hex_triplet": "0",
                "structure_id_path": "70000",
                "children": [],
            }
        ]
        flat = _flatten_structure_tree(msg)
        df = _build_structure_metadata(flat)
        annotation = np.zeros((2, 2, 2), dtype=np.uint32)
        new_vol, new_df = _remap_to_id_type(annotation, df, id_type=np.uint16)
        # After reset_index the DataFrame index is 0..N-1
        assert list(new_df.index) == [0]


# ---------------------------------------------------------------------------
# Task 2 — network download, wrapper classes, public API, mock-network tests
# ---------------------------------------------------------------------------

import sys
from unittest.mock import MagicMock, patch


def test_download_nrrd_raises_on_404(tmp_path):
    """_download_nrrd raises HTTPError on non-200 (D-03: no silent fallback)."""
    import requests as _requests

    from liom_toolkit.utils.allen_sdk import _download_nrrd

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = _requests.HTTPError("404 Not Found")
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *a: False

    dest = str(tmp_path / "out.nrrd")
    with patch("liom_toolkit.utils.allen_sdk.requests.get", return_value=mock_resp):
        with pytest.raises(_requests.HTTPError):
            _download_nrrd("http://example.com/x.nrrd", dest)


def test_download_allen_atlas_cache_hit(tmp_path):
    """Second call with cached NRRD + JSON skips download (D-03 caching contract).

    Pre-creates the annotation NRRD and structure-tree JSON in ``tmp_path`` so
    ``construct_reference_space`` reads from cache and never calls
    ``requests.get``. Does not need ``ants`` — ``construct_reference_space``
    returns the wrapper without converting to ANTsImage.
    """
    import nrrd as _nrrd

    from liom_toolkit.utils.allen_sdk import construct_reference_space

    # Build a tiny synthetic annotation volume (uint32, 2x2x2)
    annotation = np.zeros((2, 2, 2), dtype=np.uint32)
    annotation[0, 0, 0] = 997
    annotation[1, 1, 1] = 8
    nrrd_file = str(tmp_path / "allen_atlas_25.nrrd")
    _nrrd.write(nrrd_file, annotation)

    # Build a tiny synthetic structure-tree JSON
    tree_payload = {
        "msg": [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "019393",
                "structure_id_path": "997",
                "children": [
                    {
                        "id": 8,
                        "acronym": "grey",
                        "name": "Basic cell groups and regions",
                        "color_hex_triplet": "FF0000",
                        "structure_id_path": "997/8",
                        "children": [],
                    }
                ],
            }
        ]
    }
    import json as _json

    tree_file = str(tmp_path / "structure_tree.json")
    with open(tree_file, "w") as f:
        _json.dump(tree_payload, f)

    # Patch requests.get so it would raise if called (it should NOT be called)
    with patch("liom_toolkit.utils.allen_sdk.requests.get") as get_mock:
        get_mock.side_effect = AssertionError("requests.get should not be called on cache hit")
        rs = construct_reference_space(str(tmp_path), resolution=25)

    assert get_mock.call_count == 0
    assert rs.annotation.shape == (2, 2, 2)
    assert rs.structure_tree.get_structures_by_name(["root"])[0]["id"] == 997


def test_construct_reference_space_returns_wrapper(tmp_path):
    """construct_reference_space returns a wrapper with the caller-contract surface."""
    import nrrd as _nrrd

    from liom_toolkit.utils.allen_sdk import (
        _ReferenceSpace,
        _StructureTree,
        construct_reference_space,
    )

    annotation = np.zeros((2, 2, 2), dtype=np.uint32)
    annotation[0, 0, 0] = 997
    _nrrd.write(str(tmp_path / "allen_atlas_25.nrrd"), annotation)

    import json as _json

    tree_payload = {
        "msg": [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "0",
                "structure_id_path": "997",
                "children": [],
            }
        ]
    }
    with open(str(tmp_path / "structure_tree.json"), "w") as f:
        _json.dump(tree_payload, f)

    rs = construct_reference_space(str(tmp_path), resolution=25)
    assert isinstance(rs, _ReferenceSpace)
    assert isinstance(rs.structure_tree, _StructureTree)
    assert isinstance(rs.annotation, np.ndarray)
    assert callable(rs.export_itksnap_labels)
    assert callable(rs.make_structure_mask)


def test_export_itksnap_labels_via_wrapper(tmp_path):
    """The wrapper's export_itksnap_labels produces the 8-column DataFrame + volume."""
    import nrrd as _nrrd

    from liom_toolkit.utils.allen_sdk import construct_reference_space

    annotation = np.zeros((2, 2, 2), dtype=np.uint32)
    annotation[0, 0, 0] = 997
    annotation[1, 1, 1] = 8
    _nrrd.write(str(tmp_path / "allen_atlas_25.nrrd"), annotation)

    import json as _json

    tree_payload = {
        "msg": [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "019393",
                "structure_id_path": "997",
                "children": [
                    {
                        "id": 8,
                        "acronym": "grey",
                        "name": "Basic cell groups and regions",
                        "color_hex_triplet": "FF0000",
                        "structure_id_path": "997/8",
                        "children": [],
                    }
                ],
            }
        ]
    }
    with open(str(tmp_path / "structure_tree.json"), "w") as f:
        _json.dump(tree_payload, f)

    rs = construct_reference_space(str(tmp_path), resolution=25)
    vol, df = rs.export_itksnap_labels()
    # No IDX > 65535 here, so no remap — volume returned unchanged
    np.testing.assert_array_equal(vol, annotation)
    assert list(df.columns) == ["IDX", "-R-", "-G-", "-B-", "-A-", "VIS", "MSH", "LABEL"]
    assert list(df["IDX"].values) == [997, 8]


def test_make_structure_mask(tmp_path):
    """make_structure_mask builds a boolean mask for the structure + its descendants."""
    import nrrd as _nrrd

    from liom_toolkit.utils.allen_sdk import construct_reference_space

    annotation = np.zeros((2, 2, 2), dtype=np.uint32)
    annotation[0, 0, 0] = 997  # root
    annotation[0, 0, 1] = 8  # grey (descendant of root)
    annotation[1, 1, 1] = 99  # unrelated
    _nrrd.write(str(tmp_path / "allen_atlas_25.nrrd"), annotation)

    import json as _json

    tree_payload = {
        "msg": [
            {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "color_hex_triplet": "0",
                "structure_id_path": "997",
                "children": [
                    {
                        "id": 8,
                        "acronym": "grey",
                        "name": "grey",
                        "color_hex_triplet": "0",
                        "structure_id_path": "997/8",
                        "children": [],
                    }
                ],
            }
        ]
    }
    with open(str(tmp_path / "structure_tree.json"), "w") as f:
        _json.dump(tree_payload, f)

    rs = construct_reference_space(str(tmp_path), resolution=25)
    # Mask for root (id=997) should include root + its descendant grey (id=8)
    mask = rs.make_structure_mask([997])
    assert mask.dtype == bool
    assert mask[0, 0, 0]  # root voxel
    assert mask[0, 0, 1]  # grey voxel (descendant)
    assert not mask[1, 1, 1]  # unrelated voxel


def test_import_allen_sdk_no_allensdk():
    """Importing allen_sdk does not pull in allensdk (D-03: allensdk gone from runtime)."""
    # Remove any cached allensdk from a prior import, then import the module
    sys.modules.pop("allensdk", None)
    sys.modules.pop("liom_toolkit.utils.allen_sdk", None)
    import liom_toolkit.utils.allen_sdk  # noqa: F401

    assert "allensdk" not in sys.modules


def test_no_allensdk_import_in_source():
    """The allen_sdk.py source contains no allensdk import (grep-style guard)."""
    import liom_toolkit.utils.allen_sdk as mod

    source = open(mod.__file__).read()
    assert "from allensdk" not in source
    assert "import allensdk" not in source


def test_no_construct_reference_space_cache():
    """construct_reference_space_cache is deleted (allensdk gone)."""
    import liom_toolkit.utils.allen_sdk as mod

    assert not hasattr(mod, "construct_reference_space_cache")
    assert not hasattr(mod, "ReferenceSpaceCache")
    assert not hasattr(mod, "ReferenceSpace")
