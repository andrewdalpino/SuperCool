import torch

from torch import Tensor

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    SiLU,
    Upsample,
    PixelShuffle,
)

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from huggingface_hub import PyTorchModelHubMixin


class SuperCool(Module, PyTorchModelHubMixin):
    """
    A fast single-image super-resolution model with a deep low-resolution encoder network
    and high-resolution sub-pixel convolutional decoder head with global residual pathway..
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 4, 8}

    AVAILABLE_HIDDEN_RATIOS = {1, 2, 4}

    def __init__(
        self,
        upscale_ratio: int,
        num_channels: int,
        hidden_ratio: int,
        num_encoder_layers: int,
    ):
        super().__init__()

        if upscale_ratio not in self.AVAILABLE_UPSCALE_RATIOS:
            raise ValueError(
                f"Upscale ratio must be either 2, 4, or 8, {upscale_ratio} given."
            )

        if num_channels < 1:
            raise ValueError(
                f"Num channels must be greater than 0, {num_channels} given."
            )

        if hidden_ratio not in self.AVAILABLE_HIDDEN_RATIOS:
            raise ValueError(
                f"Hidden ratio must be either 1, 2, or 4, {hidden_ratio} given."
            )

        if num_encoder_layers < 1:
            raise ValueError(
                f"Num layers must be greater than 0, {num_encoder_layers} given."
            )

        self.skip = Decoder(3, upscale_ratio)

        self.encoder = Encoder(num_channels, hidden_ratio, num_encoder_layers)
        self.decoder = Decoder(num_channels, upscale_ratio)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def add_weight_norms(self) -> None:
        """Add weight normalization to all Conv2d layers in the model."""

        for module in self.modules():
            if isinstance(module, Conv2d):
                weight_norm(module)

    def remove_weight_norms(self) -> None:
        """Remove weight normalization parameterization."""

        for module in self.modules():
            if isinstance(module, Conv2d) and hasattr(module, "parametrizations"):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> Tensor:
        s = self.skip.forward(x)

        z = self.encoder.forward(x)
        z = self.decoder.forward(z)

        s += z  # Global residual connection

        return s

    @torch.no_grad()
    def upscale(self, x: Tensor) -> Tensor:
        z = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class Encoder(Module):
    def __init__(self, num_channels: int, hidden_ratio: int, num_layers: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."
        assert num_layers > 0, "Number of layers must be greater than 0."

        self.input = Conv2d(3, num_channels, kernel_size=1)

        self.body = Sequential(
            *[EncoderBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.input.forward(x)

        z = self.body.forward(z)

        return z


class EncoderBlock(Module):
    """A low-resolution encoder block with {num_channels} feature maps and wide activations."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(num_channels, hidden_channels, kernel_size=7, padding=3)
        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=7, padding=3)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        s = x.clone()

        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        s += z  # Local residual connection

        return s


class Decoder(Module):
    """A high-resolution decoder head with sub-pixel convolution and pixel shuffling."""

    def __init__(self, num_channels: int, upscale_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert upscale_ratio in {2, 4, 8}, "Upscale ratio must be either 2, 4, or 8."

        channels_out = 3 * upscale_ratio**2

        self.subpixel_conv = Conv2d(
            num_channels, channels_out, kernel_size=7, padding=3
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.subpixel_conv.forward(x)

        z = self.shuffle.forward(z)

        return z


class Bouncer(Module):
    """A residual-style discriminator network for adversarial training."""

    def __init__(self, model_size: str):
        super().__init__()

        if model_size not in {"small", "medium", "large"}:
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
            [DetectorBlock(1024, 256, 1024) for _ in range(num_tertiary_layers - 1)]
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
            ),
            SiLU(),
            Conv2d(
                in_channels=hidden_channels,
                out_channels=channels_out,
                kernel_size=1,
            ),
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
                )
            )

        self.skip = skip
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.residual(x)
        s = self.skip(x)

        z += s

        z = self.silu(z)

        return z