"""U-Net PyTorch model for vessel segmentation."""

from __future__ import annotations

import torch
import wandb
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional block for the U-Net architecture."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

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
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
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
        x = torch.cat([x, skip], axis=1)
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

        """ Encoder """
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.b = ConvBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.output_activation = nn.Sigmoid()

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
            The output tensor (sigmoid-activated segmentation map).
        """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return self.output_activation(outputs)
