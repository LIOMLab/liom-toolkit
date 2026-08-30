"""U-Net PyTorch model for vessel segmentation."""

from __future__ import annotations

# torch + wandb are moved into the [ai] extra (D-01/D-05). The upfront
# ImportError here is the honest signal on an io-only install -- the message
# names [ai] (the torch path), not [seg]. The `from e` chain preserves the
# underlying error for debugging (AGENTS §2).
try:
    import torch
    import wandb
    from torch import nn
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[ai] to use the vessel segmentation model."
    ) from e


class ConvBlock(nn.Module):
    """Convolutional block with residual connection and InstanceNorm.

    Uses InstanceNorm2d (not BatchNorm2d) — the standard for medical
    image segmentation where batch sizes are small. InstanceNorm computes
    statistics per-sample, so it works correctly at batch_size=1.

    Includes a residual connection (identity skip) when ``in_c == out_c``
    to improve gradient flow in the deep encoder/decoder paths.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_c)

        self.relu = nn.ReLU()
        # Residual connection: 1x1 conv to match channels when in_c != out_c,
        # identity skip when channels match.
        self.skip = (
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
            if in_c != out_c
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional block.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        residual = self.skip(inputs)
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.relu(x)


class EncoderBlock(nn.Module):
    """Encoder block for the U-Net architecture."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder block.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The convolved feature tensor (skip connection).
        p : torch.Tensor
            The pooled tensor passed to the next encoder stage.
        """
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):
    """Decoder block for the U-Net architecture."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor (from the previous decoder/bottleneck stage).
        skip : torch.Tensor
            The skip tensor (from the matching encoder stage).

        Returns
        -------
        torch.Tensor
            The upsampled and convolved output tensor.
        """
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VsegModel(nn.Module):
    """U-Net model for vessel segmentation."""

    def __init__(
        self,
        pretrained: bool = False,
        pretrained_artifact: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the U-Net.

        Parameters
        ----------
        pretrained : bool
            Whether to initialise from a pretrained wandb artifact. When
            ``True``, ``pretrained_artifact`` must be supplied.
        pretrained_artifact : str | None
            The wandb artifact path (``"entity/project/name:version"``) of a
            pretrained model to load. ``None`` trains from scratch. When
            ``pretrained=True`` and this is ``None``, raises ``ValueError``
            (no silent fallback to a hardcoded lab artifact — AGENTS section 2).
        device : torch.device | None
            The device to load the pretrained weights onto. ``None`` resolves
            to ``torch.device("cpu")``.

        Raises
        ------
        ValueError
            If ``pretrained=True`` and ``pretrained_artifact`` is ``None``.
        """
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        """ Encoder — 3 strided stages (matching the MONAI UNet's
        strides=(2,2,2)). Channel widths (32, 64, 128, 256) match the
        MONAI UNet configuration so the architecture comparison isolates
        the block design (residual + InstanceNorm), not the depth/width. """
        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        """ Bottleneck """
        self.b = ConvBlock(128, 256)

        """ Decoder """
        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)

        """ Classifier — outputs raw logits (no sigmoid).
        The loss function applies sigmoid internally (BCEWithLogitsLoss,
        DiceFocalLoss(sigmoid=True)) for numerical stability. The
        prediction path thresholds at sigmoid(logits) > 0.5 == logits > 0.
        """
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        if pretrained:
            if pretrained_artifact is None:
                raise ValueError(
                    "pretrained=True requires pretrained_artifact (wandb path) "
                    "— no silent fallback to a hardcoded lab artifact"
                )
            run = wandb.init()
            artifact = run.use_artifact(pretrained_artifact, type="model")
            artifact_dir = artifact.download()
            run.finish()

            checkpoint_path = artifact_dir + "/checkpoint.latest.pth"
            # PyTorch 2.6+ defaults to weights_only=True, which restricts
            # unpickling to tensors/ints/floats. Load with the safe default
            # only — the artifact is user-supplied, so the trust assumption
            # for weights_only=False no longer holds. An untrusted artifact
            # that cannot load under weights_only=True fails safe (AGENTS section 2).
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            self.load_state_dict(state)
            self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net model.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor (raw logits — apply sigmoid for probabilities).
        """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b = self.b(p3)

        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        return self.outputs(d3)
