"""sha256-verified model-artifact fetch + version registry.

This module is the integrity anchor for the shipped default nnU-Net
model. ``NnUnetV2Model`` loads checkpoints with ``weights_only=False``
-- the checkpoint is pickled code, i.e. code-execution-grade input -- so
integrity cannot come from the transport (GitHub Release assets are
mutable). Instead, a ``tag -> (url, sha256)`` registry is baked into the
installed package: the wheel commit is the trust root, and the helper
refuses to run any artifact whose digest does not match the baked-in
value.

Model <-> package version mapping policy: ``MODEL_REGISTRY`` entries are
added by release commits (``gh release upload`` attaches the zip, the
release commit bakes the entry + ``DEFAULT_MODEL_TAG``). Registry tags
need NOT equal package versions -- a retrained model adds a new entry
rather than forcing a version bump. The registry ships EMPTY: no
fabricated hashes. Artifact URLs point at GitHub Release assets of the
form ``https://github.com/LIOMLab/liom-toolkit/releases/download/<tag>/
<filename>``.

Import contract: pure stdlib (``hashlib``/``os``/``pathlib``/
``urllib.request``/``zipfile``) so ``model_v2`` can import this module at
module top without pulling torch/nnunetv2 transitively. ``file://`` URLs
work, which lets the test suite exercise the full hash-verify path with
no network.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "DEFAULT_MODEL_TAG",
    "MODEL_REGISTRY",
    "ModelArtifact",
    "default_model_dir",
    "fetch_verified",
    "model_cache_dir",
]

# Marker filename written next to a successfully verified + extracted
# model dir; its content is the registered sha256 the extraction was
# verified against.
_MARKER_NAME = ".sha256ok"

_CHUNK_SIZE = 1 << 20  # 1 MiB streaming chunks


class ModelArtifact(NamedTuple):
    """A registered model artifact: a release asset plus its integrity hash.

    Parameters
    ----------
    tag : str
        Registry key for this artifact (e.g. the GitHub Release tag).
        Tags need not equal package versions.
    url : str
        Download URL for the artifact zip (a GitHub Release asset, or a
        ``file://`` URI in tests).
    sha256 : str
        Expected lowercase hex sha256 of the zip file. This is the trust
        anchor -- the artifact is never used until its digest matches.
    filename : str
        Filename the zip is cached under inside the model cache dir.
    """

    tag: str
    url: str
    sha256: str
    filename: str


# The version -> artifact registry. Ships EMPTY: entries are baked in by
# release commits alongside the uploaded asset -- never fabricated.
MODEL_REGISTRY: dict[str, ModelArtifact] = {}

# Registry key of the artifact ``default_model_dir()`` resolves when the
# caller passes no tag. None until a release commit sets it.
DEFAULT_MODEL_TAG: str | None = None


def model_cache_dir() -> Path:
    """Return the directory downloaded model artifacts are cached under.

    Resolution order: ``$LIOM_MODEL_CACHE`` if set, else
    ``$XDG_CACHE_HOME/liom_toolkit/models`` if ``XDG_CACHE_HOME`` is set,
    else ``~/.cache/liom_toolkit/models``. The env var is the documented
    override so deployments can redirect the cache (AGENTS section 1 --
    no hardcoded lab paths).

    Returns
    -------
    Path
        The cache directory (not created by this function).
    """
    override = os.environ.get("LIOM_MODEL_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "liom_toolkit" / "models"


def fetch_verified(url: str, sha256: str, dest: str | Path) -> Path:
    """Download ``url`` to ``dest``, verifying its sha256 before the rename.

    The body is streamed to ``dest + ".partial"`` in the same directory
    while hashing; the file is renamed to ``dest`` via ``os.replace``
    ONLY after the digest matches ``sha256`` (case-insensitive). An
    interrupted download or a digest mismatch can therefore never leave
    a file at ``dest`` that a later cache check would treat as valid --
    the atomic-``.partial`` pattern from ``utils.allen_sdk`` with the
    hash check added. The ``.partial`` temp is removed on ANY exception.
    ``urllib.error.URLError`` propagates when the URL cannot be opened
    or the stream fails mid-download.

    Parameters
    ----------
    url : str
        The URL to download (``urllib.request``; ``file://`` works).
    sha256 : str
        Expected hex sha256 digest of the response body.
    dest : str | Path
        Destination filesystem path. Parent directories are created.

    Returns
    -------
    Path
        ``dest`` on success.

    Raises
    ------
    ValueError
        If the downloaded body's sha256 differs from ``sha256``. The
        message names expected vs actual digests; ``dest`` and
        ``.partial`` are absent after the raise.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    hasher = hashlib.sha256()
    try:
        # stdlib urlopen is the mandated fetch path (zero new deps);
        # permitted schemes come from the wheel-baked registry -- file://
        # is the test seam for the full verify path.
        with (
            urllib.request.urlopen(url, timeout=60) as response,  # ruff: ignore[suspicious-url-open-usage]
            partial.open("wb") as f,
        ):
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                f.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    actual = hasher.hexdigest()
    if actual.lower() != sha256.lower():
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"sha256 mismatch for {url}: expected {sha256}, got {actual}. "
            f"Refusing to install an unverified artifact at {dest}."
        )
    try:
        partial.replace(dest)  # atomic on POSIX and Windows
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dest


def default_model_dir(tag: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """Resolve, fetch, verify, and extract the registered default model.

    ``ValueError`` propagates from the fetch (sha256 mismatch) and
    extraction (member escaping the destination) steps;
    ``urllib.error.URLError`` propagates on unreachable URLs.

    Parameters
    ----------
    tag : str | None, optional
        Registry tag to resolve. ``None`` uses ``DEFAULT_MODEL_TAG``.
    cache_dir : str | Path | None, optional
        Cache root. ``None`` uses :func:`model_cache_dir`.

    Returns
    -------
    Path
        The directory containing ``dataset.json`` + ``plans.json`` (the
        nnU-Net trained-model-dir contract ``NnUnetV2Model`` validates).

    Raises
    ------
    RuntimeError
        If the registry is empty / the tag is unknown (message names the
        remediation), or if the extracted layout has no unique
        ``dataset.json`` + ``plans.json`` directory.
    """
    resolved_tag = tag if tag is not None else DEFAULT_MODEL_TAG
    entry = MODEL_REGISTRY.get(resolved_tag) if resolved_tag is not None else None
    if entry is None:
        raise RuntimeError(
            f"No model artifact registered for tag {resolved_tag!r} "
            f"(registered tags: {sorted(MODEL_REGISTRY)}). Pass an explicit "
            "model_dir, or install the release that registers an artifact."
        )

    cache = Path(cache_dir) if cache_dir is not None else model_cache_dir()
    extract_dir = cache / entry.tag
    marker = extract_dir / _MARKER_NAME

    # Cache hit: a marker recording THIS registered hash proves the
    # extracted dir was produced by a verified download. A stale or
    # foreign marker falls through to a fresh verified fetch.
    if marker.is_file() and marker.read_text().strip() == entry.sha256.lower():
        return _locate_model_root(extract_dir)

    zip_dest = cache / entry.filename
    if not _cached_zip_ok(zip_dest, entry.sha256):
        fetch_verified(entry.url, entry.sha256, zip_dest)

    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract_zip_contained(zip_dest, extract_dir)
    marker.write_text(entry.sha256.lower() + "\n")
    return _locate_model_root(extract_dir)


def _cached_zip_ok(zip_path: Path, sha256: str) -> bool:
    """Return True iff ``zip_path`` exists and its sha256 matches ``sha256``.

    A cached zip whose digest differs from the registry value is treated
    as absent -- the caller re-fetches rather than trusting a stale or
    tampered file.

    Returns
    -------
    bool
        True when the file exists and hashes to ``sha256``.
    """
    if not zip_path.is_file():
        return False
    hasher = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == sha256.lower()


def _extract_zip_contained(zip_path: Path, dest_dir: Path) -> None:
    """Extract ``zip_path`` into ``dest_dir`` with member-path containment.

    Every member's resolved target must live under ``dest_dir`` BEFORE
    any extraction -- a ``../`` member (zip-slip) raises ``ValueError``
    naming the offending member. The check is done on resolved paths so
    ``..`` components, absolute paths, and symlink-free lexical escapes
    are all caught.

    Raises
    ------
    ValueError
        If any member's resolved target escapes ``dest_dir``.
    """
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest_resolved / member.filename).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise ValueError(
                    f"zip member {member.filename!r} escapes the extraction directory {dest_dir}"
                )
        zf.extractall(dest_dir)


def _locate_model_root(extract_dir: Path) -> Path:
    """Return the directory inside ``extract_dir`` holding the nnU-Net files.

    Searches ``extract_dir`` itself, then one level deep (the export zip
    may nest the model dir). The model root is the directory containing
    both ``dataset.json`` and ``plans.json``.

    Returns
    -------
    Path
        The unique model-root directory.

    Raises
    ------
    RuntimeError
        If no candidate exists or more than one does -- the message
        names the expected layout.
    """
    candidates: list[Path] = []
    if (extract_dir / "dataset.json").is_file() and (extract_dir / "plans.json").is_file():
        candidates.append(extract_dir)
    candidates.extend(
        child
        for child in sorted(extract_dir.iterdir())
        if child.is_dir()
        and (child / "dataset.json").is_file()
        and (child / "plans.json").is_file()
    )
    if not candidates:
        raise RuntimeError(
            f"extracted model artifact at {extract_dir} contains no directory "
            "with dataset.json + plans.json (searched the root and one level "
            "deep) -- the artifact layout does not match the nnU-Net "
            "trained-model-dir contract."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"extracted model artifact at {extract_dir} is ambiguous: "
            f"{len(candidates)} directories contain dataset.json + plans.json "
            f"({[str(c) for c in candidates]}) -- expected exactly one model "
            "root."
        )
    return candidates[0]
