"""Tests for ``liom_toolkit/segmentation/vseg/artifact.py``.

Covers the sha256-verified model-artifact fetch + registry:

* ``fetch_verified`` streams a URL to ``dest + ".partial"`` while hashing,
  compares the sha256 BEFORE ``os.replace`` into place, raises
  ``ValueError`` naming expected-vs-actual on mismatch, and deletes the
  ``.partial`` on ANY error -- an unverified artifact can never be
  renamed into place (the checkpoint is ``weights_only=False`` pickled
  code, so integrity is the baked-in hash, not the transport).
* ``MODEL_REGISTRY`` starts EMPTY with ``DEFAULT_MODEL_TAG = None`` -- no
  fabricated hashes; release commits bake real entries.
* ``default_model_dir`` resolves the registry entry, downloads to
  ``<cache>/<filename>``, verifies, extracts with a member-path
  containment check, and caches the extracted dir via a ``.sha256ok``
  marker so a second call never re-downloads.
* ``model_cache_dir`` honors ``LIOM_MODEL_CACHE``.

All tests run over ``file://`` URLs (stdlib ``urllib.request`` handles
them), so the suite exercises the full hash-verify path with no network.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def model_zip(tmp_path: Path) -> tuple[Path, str]:
    """Write a minimal model zip + return ``(path, sha256_hexdigest)``.

    Layout mirrors a real exported nnU-Net model dir flattened into the
    zip: ``dataset.json``, ``plans.json``, ``fold_0/checkpoint_final.pth``
    -- the components ``default_model_dir`` locates after extraction.
    """
    zip_path = tmp_path / "liom-vseg-model.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dataset.json", '{"labels": {"background": 0, "vessel": 1}}')
        zf.writestr("plans.json", '{"plans_name": "nnUNetPlans"}')
        zf.writestr("fold_0/checkpoint_final.pth", b"stub checkpoint bytes")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return zip_path, digest


@pytest.fixture
def registered_model(
    monkeypatch: pytest.MonkeyPatch, model_zip: tuple[Path, str]
) -> tuple[Path, str]:
    """Point the registry + default tag at the ``model_zip`` fixture.

    Returns ``(zip_path, digest)`` so tests can also assert against the
    source artifact. ``MODEL_REGISTRY`` and ``DEFAULT_MODEL_TAG`` are
    restored by monkeypatch teardown.
    """
    from liom_toolkit.segmentation.vseg import artifact

    zip_path, digest = model_zip
    entry = artifact.ModelArtifact(
        tag="test-tag",
        url=zip_path.as_uri(),
        sha256=digest,
        filename=zip_path.name,
    )
    monkeypatch.setattr(artifact, "MODEL_REGISTRY", {"test-tag": entry})
    monkeypatch.setattr(artifact, "DEFAULT_MODEL_TAG", "test-tag")
    return zip_path, digest


def test_fetch_verified_happy_path(tmp_path: Path, model_zip: tuple[Path, str]) -> None:
    """A correct sha256 downloads to ``.partial``, then renames into place.

    After success the final file exists with byte-identical content and
    the ``.partial`` temp is gone -- no temp file may linger at the final
    path or beside it.
    """
    from liom_toolkit.segmentation.vseg.artifact import fetch_verified

    zip_path, digest = model_zip
    dest = tmp_path / "out" / "model.zip"

    result = fetch_verified(zip_path.as_uri(), digest, dest)

    assert result == dest
    assert dest.is_file()
    assert dest.read_bytes() == zip_path.read_bytes()
    assert not (dest.parent / (dest.name + ".partial")).exists()


def test_fetch_verified_hash_mismatch_raises_and_cleans(tmp_path: Path, model_zip) -> None:
    """A wrong sha256 raises ValueError naming expected vs actual.

    The mismatch path must leave NEITHER the destination nor the
    ``.partial`` behind: an unverified artifact can never masquerade as a
    complete download (the checkpoint is ``weights_only=False`` pickled
    code -- a tampered zip is code-execution-grade input).
    """
    from liom_toolkit.segmentation.vseg.artifact import fetch_verified

    zip_path, digest = model_zip
    wrong = "0" * 64
    dest = tmp_path / "model.zip"

    with pytest.raises(ValueError, match=wrong) as excinfo:
        fetch_verified(zip_path.as_uri(), wrong, dest)

    assert digest in str(excinfo.value)  # actual digest named too
    assert not dest.exists()
    assert not (dest.parent / (dest.name + ".partial")).exists()


def test_fetch_verified_unreachable_url_leaves_nothing(tmp_path: Path) -> None:
    """An unreachable ``file://`` URL surfaces URLError, never a fabricated file.

    A failed open must propagate as a URLError-class exception and leave
    no ``dest`` / ``.partial`` -- a missing artifact is an explicit
    failure, not a zero-byte file at the final path.
    """
    from urllib.error import URLError

    from liom_toolkit.segmentation.vseg.artifact import fetch_verified

    missing = tmp_path / "does-not-exist.zip"
    dest = tmp_path / "model.zip"

    with pytest.raises(URLError):
        fetch_verified(missing.as_uri(), "0" * 64, dest)

    assert not dest.exists()
    assert not (dest.parent / (dest.name + ".partial")).exists()


def test_default_model_dir_empty_registry_raises_remediation(tmp_path: Path) -> None:
    """An empty registry raises RuntimeError naming the remediation.

    Until a release commit bakes an entry, ``default_model_dir`` cannot
    fabricate a model -- it must fail loudly and tell the caller to pass
    an explicit ``model_dir`` or install the release that registers an
    artifact.
    """
    from liom_toolkit.segmentation.vseg.artifact import default_model_dir

    with pytest.raises(RuntimeError, match="model_dir"):
        default_model_dir(cache_dir=tmp_path / "cache")


def test_default_model_dir_fetches_extracts_and_caches(
    tmp_path: Path, registered_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First call downloads + extracts; second call reuses the ``.sha256ok`` cache.

    The returned directory contains ``dataset.json`` + ``plans.json``
    (the nnU-Net model-dir contract). A second call must NOT re-open the
    URL -- the marker proving the extracted dir matches the registered
    hash is the cache hit.
    """
    import urllib.request

    from liom_toolkit.segmentation.vseg import artifact

    cache_dir = tmp_path / "cache"

    model_dir = artifact.default_model_dir(cache_dir=cache_dir)

    assert (model_dir / "dataset.json").is_file()
    assert (model_dir / "plans.json").is_file()
    assert (model_dir / "fold_0" / "checkpoint_final.pth").is_file()

    # Second call: any urlopen would mean the cache marker was ignored.
    calls = []
    real_urlopen = urllib.request.urlopen  # ruff: ignore[suspicious-url-open-usage] -- delegating wrapper, scheme is file:// here

    def counting_urlopen(*args, **kwargs):
        calls.append(args)
        return real_urlopen(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", counting_urlopen)

    again = artifact.default_model_dir(cache_dir=cache_dir)

    assert again == model_dir
    assert calls == []


def test_default_model_dir_stale_marker_reextracts(
    tmp_path: Path, registered_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale ``.sha256ok`` marker re-extracts from the verified cached zip.

    The marker is only trustworthy when it records the CURRENT registered
    hash -- a stale or foreign marker must not be treated as a cache hit.
    But a still-valid cached zip does not need re-downloading: the second
    call re-extracts (restoring a deleted member and rewriting the
    marker) WITHOUT re-opening the URL.
    """
    import urllib.request

    from liom_toolkit.segmentation.vseg import artifact

    cache_dir = tmp_path / "cache"
    model_dir = artifact.default_model_dir(cache_dir=cache_dir)

    # Simulate a stale/foreign marker plus a tampered extracted tree.
    (model_dir / ".sha256ok").write_text("bogus-hash\n")
    (model_dir / "fold_0" / "checkpoint_final.pth").unlink()

    calls = []
    real_urlopen = urllib.request.urlopen  # ruff: ignore[suspicious-url-open-usage] -- delegating wrapper, scheme is file:// here

    def counting_urlopen(*args, **kwargs):
        calls.append(args)
        return real_urlopen(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", counting_urlopen)

    again = artifact.default_model_dir(cache_dir=cache_dir)

    assert again == model_dir
    assert calls == []  # valid cached zip -> re-extract, no re-download
    assert (model_dir / "fold_0" / "checkpoint_final.pth").is_file()
    assert (model_dir / ".sha256ok").read_text().strip() == registered_model[1]


def test_default_model_dir_corrupt_cached_zip_refetches(
    tmp_path: Path, registered_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached zip whose hash differs from the registry is re-fetched.

    The cached zip is trusted only while its digest matches the baked-in
    registry value -- a corrupted or replaced cache file triggers a fresh
    verified download rather than being extracted blind. (The extracted
    dir's ``.sha256ok`` marker is removed too so the else branch runs at
    all -- a valid marker alone is already a cache hit.)
    """
    import urllib.request

    from liom_toolkit.segmentation.vseg import artifact

    cache_dir = tmp_path / "cache"
    zip_path, _digest = registered_model
    model_dir = artifact.default_model_dir(cache_dir=cache_dir)

    # Corrupt the cached zip and invalidate the marker.
    cached_zip = cache_dir / zip_path.name
    cached_zip.write_bytes(b"corrupted")
    (model_dir / ".sha256ok").unlink()

    calls = []
    real_urlopen = urllib.request.urlopen  # ruff: ignore[suspicious-url-open-usage] -- delegating wrapper, scheme is file:// here

    def counting_urlopen(*args, **kwargs):
        calls.append(args)
        return real_urlopen(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", counting_urlopen)

    model_dir = artifact.default_model_dir(cache_dir=cache_dir)

    assert len(calls) == 1
    assert (model_dir / "dataset.json").is_file()


def test_extract_rejects_path_traversal_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zip member escaping the destination raises ValueError.

    A ``../`` member would write outside the cache dir (classic zip-slip);
    the extraction must check every member's resolved path against the
    destination BEFORE writing, and fail naming the offending member.
    """
    from liom_toolkit.segmentation.vseg import artifact

    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("dataset.json", "{}")
        zf.writestr("plans.json", "{}")
        zf.writestr("../evil.txt", "pwned")
    evil_digest = hashlib.sha256(evil_zip.read_bytes()).hexdigest()

    entry = artifact.ModelArtifact(
        tag="evil-tag",
        url=evil_zip.as_uri(),
        sha256=evil_digest,
        filename=evil_zip.name,
    )
    monkeypatch.setattr(artifact, "MODEL_REGISTRY", {"evil-tag": entry})
    monkeypatch.setattr(artifact, "DEFAULT_MODEL_TAG", "evil-tag")

    with pytest.raises(ValueError, match=r"\.\./evil\.txt"):
        artifact.default_model_dir(cache_dir=tmp_path / "cache")


def test_model_cache_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LIOM_MODEL_CACHE`` overrides the default cache location.

    The cache dir is a documented parameter-with-default (AGENTS section
    1 -- no hardcoded lab paths); the env var lets deployments redirect it.
    """
    from liom_toolkit.segmentation.vseg.artifact import model_cache_dir

    monkeypatch.setenv("LIOM_MODEL_CACHE", str(tmp_path / "custom-cache"))

    assert model_cache_dir() == tmp_path / "custom-cache"


def test_registry_starts_empty_with_no_default_tag() -> None:
    """``MODEL_REGISTRY`` ships empty and ``DEFAULT_MODEL_TAG`` is None.

    No fabricated hashes are baked in -- entries are added by release
    commits that also compute the real artifact digest. This guards the
    contract at import time so a stray entry cannot sneak in unnoticed.
    """
    from liom_toolkit.segmentation.vseg import artifact

    assert artifact.MODEL_REGISTRY == {}
    assert artifact.DEFAULT_MODEL_TAG is None
