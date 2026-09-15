"""nnU-Net v2 predictor wrapper: :class:`NnUnetV2Model`.

This module adapts nnU-Net v2's ``nnUNetPredictor`` (from
``nnunetv2.inference.predict_from_raw_data``) to the toolkit's array +
spacing contract. The wrapper owns the predictor, plans, and checkpoint as a
unit via ``initialize_from_trained_model_folder`` and turns a ``(C, Z, H,
W)`` array plus per-axis spacing into a 0/255 uint8 vessel mask -- the
single class every nnU-Net inference surface routes through.

Security note: ``model_dir`` must be a TRUSTED nnU-Net training output
directory. ``initialize_from_trained_model_folder`` loads checkpoints with
``torch.load(weights_only=False)`` internally -- the checkpoint is pickled
code -- so the directory contents must come from a trusted source (your own
``nnUNetv2_train`` output or a verified artifact).

Import contract: torch is guarded at module top (the ``[ai]`` extra);
``nnunetv2`` is imported function-scope so this module loads with only torch
installed. The wrapper is NOT a ``torch.nn.Module`` -- the predictor owns the
full inference pipeline (preprocessing, sliding window, ensembling); this
class only adapts array and spacing contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an install without the extra (mirrors the ssl/warmstart.py
# guard). The `from e` chain preserves the underlying error. nnunetv2 is
# imported function-scope so this module loads with only torch installed.
try:
    import torch
except ImportError as e:  # pragma: no cover - exercised only on installs without [ai]
    raise ImportError(
        "Please install liom-toolkit[ai] to use the nnU-Net vessel segmentation model."
    ) from e

if TYPE_CHECKING:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

__all__ = ["NnUnetV2Model"]


class NnUnetV2Model:
    """Wrapper around nnU-Net v2's ``nnUNetPredictor`` for vessel segmentation.

    Construction is eager: ``__init__`` validates the ``model_dir`` contract
    (a directory containing ``dataset.json``, ``plans.json``, and at least
    one usable ``fold_*/<checkpoint_name>``) BEFORE constructing the
    predictor, then initializes network + plans + checkpoint via
    ``initialize_from_trained_model_folder`` in one call. A bogus model_dir
    therefore fails at construction time with a path-naming ``ValueError``,
    not mid-inference with nnU-Net's opaque FileNotFoundError.

    Parameters
    ----------
    model_dir : str | Path
        Path to a trained-model output directory (the
        ``<DatasetName>/<Trainer>__<PlansIdentifier>__<configuration>/``
        directory produced by ``nnUNetv2_train``). Must contain
        ``dataset.json``, ``plans.json``, and ``fold_<n>/<checkpoint_name>``.
        Must be a TRUSTED artifact -- upstream loads checkpoints with
        ``weights_only=False``.
    device : torch.device | str | None, optional
        Compute device. ``None`` (default) resolves to
        ``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` --
        nnU-Net's constructor defaults to ``cuda`` unconditionally, so the
        wrapper always resolves and passes the device explicitly.
        ``perform_everything_on_device`` tracks ``device.type == "cuda"``.
    use_folds : tuple[int | str, ...] | None, optional
        Folds to ensemble. ``None`` defers to nnU-Net's auto-detection
        (every ``fold_<n>`` containing the checkpoint, excluding
        ``fold_all``). When given, exactly those fold directories are
        validated up front; ``"all"`` selects ``fold_all``. The annotation
        is the public contract -- a variadic tuple of fold ints and/or the
        literal ``"all"``; at runtime a bare scalar or list is normalized
        to a tuple for convenience (upstream's
        ``initialize_from_trained_model_folder`` likewise wraps a bare
        ``str`` in a list).
    checkpoint_name : str, optional
        Checkpoint filename inside each fold directory. Defaults to
        ``"checkpoint_final.pth"``.
    tile_step_size : float, optional
        Sliding-window tile step as a fraction of the patch size. Must be
        in the interval ``(0, 1]`` -- nnU-Net's
        ``compute_steps_for_sliding_window`` steps by
        ``tile_step_size * tile``, so a value ``> 1`` leaves un-predicted
        gaps the Gaussian blending fills with near-zero-weight garbage,
        and ``<= 0`` crashes on division upstream. Defaults to 0.5.
    use_gaussian : bool, optional
        Weighted Gaussian blending across overlapping tiles. Defaults to
        True.
    use_mirroring : bool, optional
        Test-time mirroring augmentation. Defaults to True.
    allow_tqdm : bool, optional
        Forwarded to the predictor's progress bars. Defaults to False.
    verbose : bool, optional
        Predictor verbosity (also drives ``verbose_preprocessing``).
        Defaults to False.

    Attributes
    ----------
    predictor : nnUNetPredictor
        The owned, initialized nnU-Net predictor.
    device : torch.device
        The resolved compute device.
    model_dir : Path
        The validated model directory.

    Raises
    ------
    ValueError
        If ``model_dir`` is not a directory, or lacks ``dataset.json``,
        ``plans.json``, or a usable ``fold_*/<checkpoint_name>``, or if
        ``tile_step_size`` is outside ``(0, 1]``. The message names the
        offending path component or value.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device | str | None = None,
        use_folds: tuple[int | str, ...] | None = None,
        checkpoint_name: str = "checkpoint_final.pth",
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        allow_tqdm: bool = False,
        verbose: bool = False,
    ) -> None:
        # nnUNetPredictor stores tile_step_size verbatim with no bounds
        # check; the wrapper validates the (0, 1] contract up front -- a
        # step > 1 leaves uncovered image regions that the Gaussian
        # blending renders as plausible-but-wrong output.
        if not 0 < tile_step_size <= 1:
            raise ValueError(f"tile_step_size must be in the interval (0, 1]; got {tile_step_size}")

        model_dir = Path(model_dir)
        self._validate_model_dir(model_dir, use_folds, checkpoint_name)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # nnunetv2 is in the [ai] extra -- import function-scope so this
        # module loads with only torch installed.
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError as e:  # pragma: no cover - exercised only without nnunetv2
            raise ImportError(
                "Please install liom-toolkit[ai] to use the nnU-Net vessel "
                "segmentation model (nnunetv2 missing)."
            ) from e

        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=device.type == "cuda",
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm,
        )
        predictor.initialize_from_trained_model_folder(str(model_dir), use_folds, checkpoint_name)
        self.predictor: nnUNetPredictor = predictor
        self.device: torch.device = device
        self.model_dir: Path = model_dir

    @staticmethod
    def _validate_model_dir(
        model_dir: Path,
        use_folds: tuple[int | str, ...] | None,
        checkpoint_name: str,
    ) -> None:
        """Validate the trained-model directory contract before touching nnU-Net.

        Checks, in order: ``model_dir`` is a directory, ``dataset.json``
        exists, ``plans.json`` exists, and at least one usable fold
        checkpoint exists. With ``use_folds=None`` the check mirrors nnU-
        Net's ``auto_detect_available_folds`` (every ``fold_<n>`` containing
        the checkpoint counts; ``fold_all`` is excluded). With explicit
        ``use_folds``, exactly those ``fold_<f>/<checkpoint_name>`` files are
        required.

        Raises
        ------
        ValueError
            On the first contract violation; the message names the offending
            path or component.
        """
        if not model_dir.is_dir():
            raise ValueError(
                f"model_dir is not a directory: {model_dir} -- expected a "
                f"trained-model output directory containing dataset.json, "
                f"plans.json, and fold_<n>/{checkpoint_name}"
            )
        if not (model_dir / "dataset.json").is_file():
            raise ValueError(
                f"model_dir lacks dataset.json: {model_dir / 'dataset.json'} does not exist"
            )
        if not (model_dir / "plans.json").is_file():
            raise ValueError(
                f"model_dir lacks plans.json: {model_dir / 'plans.json'} does not exist"
            )
        if use_folds is not None:
            folds = use_folds if isinstance(use_folds, (tuple, list)) else (use_folds,)
            # An empty sequence passes the loop vacuously, then nnU-Net
            # loads zero checkpoints -- list_of_parameters stays empty and
            # inference crashes on `None.to('cpu')` deep in the predictor.
            if len(folds) == 0:
                raise ValueError(
                    "use_folds must be None or a non-empty sequence of fold "
                    "indices; got an empty sequence"
                )
            for fold in folds:
                checkpoint = model_dir / f"fold_{fold}" / checkpoint_name
                if not checkpoint.is_file():
                    raise ValueError(
                        f"fold checkpoint not found: {checkpoint} -- "
                        f"use_folds={tuple(folds)} requires each "
                        f"fold_<f>/{checkpoint_name} to exist"
                    )
        else:
            usable = [
                fold_dir
                for fold_dir in model_dir.glob("fold_*")
                if fold_dir.is_dir()
                and fold_dir.name != "fold_all"
                and (fold_dir / checkpoint_name).is_file()
            ]
            if not usable:
                raise ValueError(
                    f"no usable fold checkpoint under {model_dir}: expected "
                    f"at least one fold_<n>/{checkpoint_name} (fold_all is "
                    f"excluded from auto-detection; pass use_folds=('all',) "
                    f"to request it)"
                )

    def predict_proba(self, arr: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
        """Return per-class probabilities for ``arr`` via the nnU-Net pipeline.

        The array is passed to ``predict_single_npy_array`` cast to float32
        and otherwise unmodified -- nnU-Net normalizes per its plans, so no
        CLAHE / min-max preprocessing happens here. ``image_properties``
        carries only the ``'spacing'`` key, in the array's own spatial axis
        order.

        Parameters
        ----------
        arr : np.ndarray
            Input array, ``(C, Z, H, W)`` -- channel first, spatial axes in
            z,y,x order. A single 2D slice must be promoted to
            ``(1, 1, H, W)`` by the caller: every nnunetv2 preprocessor
            applies ``plans_manager.transpose_forward``, which is always
            length 3 (``ExperimentPlanner.determine_transpose`` is hardcoded
            to ``range(3)`` -- even 2D readers emit ``(c, 1, X, Y)`` with
            3-element spacing), so ``run_case_npy``'s 4-element transpose
            permutation crashes on a 3D input deep inside the dependency.
        spacing : tuple[float, ...]
            Per-spatial-axis spacing in z,y,x order; ``len(spacing) == 3``.
            Required (no default) because a silently-defaulted spacing
            mis-resamples the volume. For a promoted 2D slice the
            through-plane value is a pass-through for a ``'2d'``
            configuration (``default_preprocessor`` keeps
            ``original_spacing[0]`` verbatim); nnU-Net's own
            ``NaturalImage2DIO`` reports 999 for it.

        Returns
        -------
        np.ndarray
            Probability array, ``(num_classes, *spatial)`` -- the softmax
            output, not the labelmap.

        Raises
        ------
        ValueError
            If ``arr`` is not an ndarray, ``arr.ndim`` is not 4, a channel
            or spatial dim is empty, ``arr`` contains NaN or infinite
            values, ``len(spacing)`` is not 3, or a spacing value is
            non-finite or <= 0.
        """
        if not isinstance(arr, np.ndarray):
            # ValueError, not TypeError: the wrapper's contract is that every
            # invalid input surfaces as ValueError naming the offending value.
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                f"arr must be a numpy ndarray, got {type(arr).__name__}"
            )
        if arr.ndim != 4:
            raise ValueError(
                f"arr.ndim must be 4 (C,Z,H,W); got ndim={arr.ndim} with "
                f"shape {arr.shape} -- promote a single 2D slice to "
                f"(1, 1, H, W) before calling"
            )
        if arr.shape[0] < 1:
            raise ValueError(f"arr must have at least one channel; got shape {arr.shape}")
        for axis, size in enumerate(arr.shape[1:], start=1):
            if size < 1:
                raise ValueError(f"spatial axis {axis} of arr is empty (size 0); shape {arr.shape}")
        try:
            spacing = tuple(spacing)
        except TypeError:
            raise ValueError(
                f"spacing must be a sequence of per-axis values; got {spacing!r}"
            ) from None
        if len(spacing) != 3:
            raise ValueError(
                f"len(spacing) must be 3 (one per z,y,x spatial axis of the "
                f"(C,Z,H,W) input -- the always-length-3 transpose_forward "
                f"permutation indexes spacing[0..2]); got "
                f"len(spacing)={len(spacing)} ({spacing})"
            )
        for value in spacing:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"spacing values must be finite and > 0; got {value} in spacing={spacing}"
                )
        arr_f32 = np.asarray(arr, dtype=np.float32)
        # Screen on the float32 array actually fed to the model: this also
        # catches float64 -> float32 overflow to inf. nnU-Net's normalization
        # propagates NaN/inf through the network into a plausible
        # all-background mask -- a silent wrong-result path, so fail here.
        if not np.isfinite(arr_f32).all():
            raise ValueError(
                "arr contains NaN or infinite values -- drop or replace "
                "non-finite voxels before calling predict_proba"
            )
        _, probs = self.predictor.predict_single_npy_array(
            arr_f32,
            {"spacing": list(spacing)},
            save_or_return_probabilities=True,
        )
        return probs

    def predict(self, arr: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
        """Return the binary vessel mask for ``arr`` as 0/255 uint8.

        Channel 1 of the 2-class softmax IS the vessel probability; the mask
        is ``(probs[1] > 0.5) * 255`` with a strict threshold (exactly 0.5
        maps to 0). The output is positionally identical to the input's
        spatial dims -- no reordering, no sorting.

        Parameters
        ----------
        arr : np.ndarray
            Input array, ``(C, Z, H, W)`` -- a single 2D slice is passed as
            ``(1, 1, H, W)`` and its ``(1, H, W)`` output squeezed back by
            the caller.
        spacing : tuple[float, ...]
            Per-spatial-axis spacing in z,y,x order; ``len(spacing) == 3``.

        Returns
        -------
        np.ndarray
            ``uint8`` mask with values in {0, 255}, shape ``arr.shape[1:]``.

        Raises
        ------
        ValueError
            Propagated from :meth:`predict_proba` input validation, and when
            the model returns fewer than 2 probability channels -- a
            single-class softmax has no vessel channel, so reading
            ``probs[1]`` would be a silent wrong-channel read.
        """
        probs = self.predict_proba(arr, spacing)
        if probs.shape[0] < 2:
            raise ValueError(
                f"predict_proba returned {probs.shape[0]} probability "
                f"channel(s); the binary vessel contract requires at least "
                f"2 (channel 1 is the vessel probability)"
            )
        return (probs[1] > 0.5).astype(np.uint8) * 255
