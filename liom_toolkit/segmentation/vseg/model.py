import torch
import torch.nn as nn
import wandb


class ConvBlock(nn.Module):
    """
    Convolutional block for the U-Net architecture
    """

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block

        :param inputs: The input tensor
        :type inputs: torch.Tensor
        :return: The output tensor
        :rtype: torch.Tensor
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block for the U-Net architecture
    """

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the encoder block

        :param inputs: The input tensor
        :type inputs: torch.Tensor
        :return: The output tensor
        :rtype: torch.Tensor
        """
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):
    """
    Decoder block for the U-Net architecture
    """

    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block

        :param inputs: The input tensor
        :type inputs: torch.Tensor
        :param skip: The skip tensor
        :type skip: torch.Tensor
        :return: The output tensor
        :rtype: torch.Tensor
        """
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class VsegModel(nn.Module):
    """
    U-Net model for vessel segmentation
    """

    def __init__(self, pretrained: bool = False, device: torch.device = torch.device('cpu')):
        super().__init__()

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
            run = wandb.init()
            artifact = run.use_artifact('liom-lab/model-registry/Vessel Segmentation:latest', type='model')
            artifact_dir = artifact.download()
            run.finish()

            state = torch.load(artifact_dir + "/checkpoint.latest.pth", map_location=device)
            self.load_state_dict(state)
            self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model

        :param inputs: The input tensor
        :type inputs: torch.Tensor
        :return: The output tensor
        :rtype: torch.Tensor
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
        outputs = self.output_activation(outputs)
        return outputs
