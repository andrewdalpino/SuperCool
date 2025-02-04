import torch

from torch import Tensor

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    Sigmoid,
    Upsample,
    PixelShuffle,
    BatchNorm2d,
    SiLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Identity,
    Flatten,
    Linear,
    Parameter,
)

from torch.nn.utils.parametrizations import weight_norm

from huggingface_hub import PyTorchModelHubMixin


class SuperCool(Module, PyTorchModelHubMixin):
    """
    A fast single-image super-resolution model with a deep low-resolution encoder network
    and high-resolution sub-pixel convolutional decoder head with global residual pathway.
    """

    def __init__(
        self,
        base_upscaler: str,
        upscale_ratio: int,
        num_channels: int,
        hidden_ratio: int,
        num_layers: int,
    ):
        super().__init__()

        if base_upscaler not in ("bilinear", "bicubic"):
            raise ValueError(
                f"Base upscaler must be bilinear or bicubic, {base_upscaler} given."
            )

        if upscale_ratio not in (2, 4, 8):
            raise ValueError(
                f"Upscale ratio must be either 2, 4, or 8, {upscale_ratio} given."
            )

        if num_channels < 1:
            raise ValueError(
                f"Num channels must be greater than 0, {num_channels} given."
            )

        if hidden_ratio not in (1, 2, 4):
            raise ValueError(
                f"Hidden ratio must be either 1, 2, or 4, {hidden_ratio} given."
            )

        if num_layers < 1:
            raise ValueError(f"Num layers must be greater than 0, {num_layers} given.")

        self.input = weight_norm(Conv2d(3, num_channels, kernel_size=5, padding=2))

        self.skip = Upsample(scale_factor=upscale_ratio, mode=base_upscaler)

        self.encoder = Sequential(
            *[EncoderBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
        )

        self.decoder = SubpixelConv2d(
            num_channels, upscale_ratio, kernel_size=3, padding=1
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        z = self.input(x)

        z = self.encoder(z)

        z = self.decoder(z)
        z = self.shuffle(z)

        s = self.skip(x)

        z += s  # Global residual connection

        z = torch.clamp(z, 0, 1)

        return z


class EncoderBlock(Module):
    """A low-resolution encoder block with {num_channels} feature maps and wide activations."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        if hidden_ratio not in (1, 2, 4):
            raise ValueError(
                f"Hidden ratio must be either 1, 2, or 4, {hidden_ratio} given."
            )

        hidden_channels = num_channels * hidden_ratio

        conv1 = Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1)
        conv2 = Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.conv1 = weight_norm(conv1)
        self.conv2 = weight_norm(conv2)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1(x)
        z = self.silu(z)
        z = self.conv2(z)

        z += x  # Local residual connection

        return z


class SubpixelConv2d(Module):
    """A sub-pixel (1 / upscale_ratio) convolution layer with weight normalization."""

    def __init__(
        self, num_channels: int, upscale_ratio: int, kernel_size: int, padding: int
    ):
        super().__init__()

        channels_out = 3 * upscale_ratio**2

        conv = Conv2d(
            num_channels, channels_out, kernel_size=kernel_size, padding=padding
        )

        self.conv = weight_norm(conv)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Bouncer(Module):
    """A residual-style discriminator network for adversarial training."""

    def __init__(self, model_size: str):
        super().__init__()

        if model_size not in ("small", "medium", "large"):
            raise ValueError(
                f"Model size must be small, medium, or large, {model_size} given."
            )

        num_primary_layers = 3
        num_quaternary_layers = 3

        match model_size:
            case "small":
                num_secondary_layers = 4
                num_tertiary_layers = 6
            case "medium":
                num_secondary_layers = 4
                num_tertiary_layers = 23
            case "large":
                num_secondary_layers = 8
                num_tertiary_layers = 36

        self.input = Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            SiLU(),
            MaxPool2d(3, stride=2, padding=1),
        )

        self.detector = Sequential(DetectorBlock(64, 64, 256))

        self.detector.extend(
            [DetectorBlock(256, 64, 256) for _ in range(num_primary_layers - 1)]
        )

        self.detector.append(DetectorBlock(256, 128, 512, stride=2))

        self.detector.extend(
            [DetectorBlock(512, 128, 512) for _ in range(num_secondary_layers - 1)]
        )

        self.detector.append(DetectorBlock(512, 256, 1024, stride=2))

        self.detector.extend(
            [DetectorBlock(512, 256, 1024) for _ in range(num_tertiary_layers - 1)]
        )

        self.detector.append(DetectorBlock(1024, 512, 2048, stride=2))

        self.detector.extend(
            [DetectorBlock(2048, 512, 2048) for _ in range(num_quaternary_layers - 1)]
        )

        self.pool = AdaptiveAvgPool2d(1)

        self.flatten = Flatten(start_dim=1)

        self.linear = Linear(2048, 1)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)

        x = self.detector(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x


class DetectorBlock(Module):
    """A residual bottleneck block with 3x3 convolutions and batch normalization."""

    def __init__(
        self, channels_in: int, hidden_channels: int, channels_out: int, stride: int = 1
    ):
        super().__init__()

        self.residual = Sequential(
            Conv2d(
                in_channels=channels_in,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            BatchNorm2d(hidden_channels),
            SiLU(),
            Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(hidden_channels),
            SiLU(),
            Conv2d(
                in_channels=hidden_channels,
                out_channels=channels_out,
                kernel_size=1,
                bias=False,
            ),
            BatchNorm2d(channels_out),
        )

        if channels_in == channels_out:
            skip = Sequential(Identity())
        else:
            skip = Sequential(
                Conv2d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(channels_out),
            )

        self.skip = skip
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.residual(x)
        s = self.skip(x)

        z += s

        z = self.silu(z)

        return z
