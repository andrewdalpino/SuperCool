from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Conv2d,
    Linear,
    Sigmoid,
    SiLU,
    RMSNorm,
    AvgPool2d,
    AdaptiveAvgPool2d,
    PixelShuffle,
    Flatten,
)

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class SuperCool(Module, PyTorchModelHubMixin):
    """
    A single-image super-resolution model with a deep low-resolution residual encoder network
    and high-resolution sub-pixel convolution decoder head with global residual pathway.
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 3, 4}

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
                f"Upscale ratio must be either 2, 3, or 4, {upscale_ratio} given."
            )

        self.skip = SubpixelConv2d(3, upscale_ratio)

        self.encoder = Encoder(num_channels, hidden_ratio, num_encoder_layers)

        self.decoder = SubpixelConv2d(num_channels, upscale_ratio)

        self.upscale_ratio = upscale_ratio

    @property
    def num_params(self) -> int:
        """Total number of parameters in the model."""

        return sum(param.numel() for param in self.parameters())

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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s = self.skip.forward(x)

        z = self.encoder.forward(x)
        z = self.decoder.forward(z)

        z = s + z  # Global residual connection

        return z, s

    @torch.no_grad()
    def test_compare(self, x: Tensor) -> Tensor:
        z, s = self.forward(x)

        z = torch.clamp(z, 0, 1)
        s = torch.clamp(s, 0, 1)

        return z, s

    @torch.no_grad()
    def upscale(self, x: Tensor) -> Tensor:
        z, _ = self.forward(x)

        z = torch.clamp(z, 0, 1)

        z *= 255

        z = z.to(torch.uint8)

        return z


class Encoder(Module):
    """A low-resolution subnetwork employing a deep stack of encoder blocks."""

    def __init__(self, num_channels: int, hidden_ratio: int, num_layers: int):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0."

        self.stem = Conv2d(3, num_channels, kernel_size=3, padding=1)

        self.body = ModuleList(
            [EncoderBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
        )

        self.checkpoint = lambda layer, x: layer(x)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem.forward(x)

        for layer in self.body:
            z = self.checkpoint(layer, z)

        return z


class EncoderBlock(Module):
    """A single encoder block consisting of three stages and a residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.stage1 = SpatialAttention(num_channels)
        self.stage2 = InvertedBottleneck(num_channels, hidden_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stage1.forward(x)
        z = self.stage2.forward(z)

        z = x + z  # Local residual connection

        return z


class SpatialAttention(Module):
    """A spatial attention module with large depth-wise convolutions."""

    def __init__(self, num_channels: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        self.depthwise = Conv2d(
            num_channels,
            num_channels,
            kernel_size=11,
            padding=5,
            groups=num_channels,
            bias=False,
        )

        self.pointwise = Conv2d(num_channels, num_channels, kernel_size=1)

        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        z = self.depthwise.forward(x)
        z = self.pointwise.forward(z)

        z = self.sigmoid.forward(z)

        z = z * x

        return z


class InvertedBottleneck(Module):
    """A wide non-linear activation block with 3x3 convolutions."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        return z


class SubpixelConv2d(Module):
    """A high-resolution decoder using sub-pixel convolution."""

    def __init__(self, in_channels: int, upscale_ratio: int):
        super().__init__()

        assert upscale_ratio in {2, 3, 4}, "Upscale ratio must be either 2, 3, or 4."

        out_channels = 3 * upscale_ratio**2

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.shuffle = PixelShuffle(upscale_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.shuffle.forward(z)

        return z


class Bouncer(Module):
    """A discriminator network for adversarial training."""

    def __init__(self):
        super().__init__()

        self.detector = Detector(
            num_primary_channels=64,
            num_primary_layers=3,
            num_secondary_channels=128,
            num_secondary_layers=3,
            num_tertiary_channels=256,
            num_tertiary_layers=6,
            num_quaternary_channels=512,
            num_quaternary_layers=3,
        )

        self.pool = AdaptiveAvgPool2d(1)

        self.flatten = Flatten(start_dim=1)

        self.classifier = BinaryClassifier(512)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        z = self.detector.forward(x)

        z = self.pool.forward(z)
        z = self.flatten.forward(z)

        z = self.classifier.forward(z)

        return z


class Detector(Module):
    """A feature extractor for the discriminator network."""

    def __init__(
        self,
        num_primary_channels: int,
        num_primary_layers: int,
        num_secondary_channels: int,
        num_secondary_layers: int,
        num_tertiary_channels: int,
        num_tertiary_layers: int,
        num_quaternary_channels: int,
        num_quaternary_layers: int,
    ):
        super().__init__()

        assert (
            num_primary_layers > 0
        ), "Number of primary layers must be greater than 0."
        assert (
            num_secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."
        assert (
            num_tertiary_layers > 0
        ), "Number of tertiary layers must be greater than 0."
        assert (
            num_quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.stem = Conv2d(3, num_primary_channels, kernel_size=4, stride=4)

        body = Sequential()

        body.extend(
            [DetectorBlock(num_primary_channels) for _ in range(num_primary_layers)]
        )

        body.append(AvgPool2d(kernel_size=2, stride=2))

        body.extend(
            [DetectorBlock(num_secondary_channels) for _ in range(num_secondary_layers)]
        )

        body.append(AvgPool2d(kernel_size=2, stride=2))

        body.extend(
            [DetectorBlock(num_tertiary_channels) for _ in range(num_tertiary_layers)]
        )

        body.append(AvgPool2d(kernel_size=2, stride=2))

        body.extend(
            [
                DetectorBlock(num_quaternary_channels)
                for _ in range(num_quaternary_layers)
            ]
        )

        self.body = body

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem.forward(x)
        z = self.body.forward(z)

        return z


class DetectorBlock(Module):
    """A detector block with depth-wise separable convolution and residual connection."""

    def __init__(self, num_channels: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        hidden_channels = 4 * num_channels

        self.conv1 = Conv2d(
            num_channels,
            num_channels,
            kernel_size=7,
            padding=3,
            num_groups=num_channels,
            bias=False,
        )

        self.conv2 = Conv2d(num_channels, hidden_channels, kernel_size=1)
        self.conv3 = Conv2d(hidden_channels, num_channels, kernel_size=1)

        self.norm = RMSNorm(num_channels)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm.forward(x)
        z = self.conv1.forward(z)
        z = self.conv2.forward(z)
        z = self.silu.forward(z)
        z = self.conv3.forward(z)

        z = x + z  # Local residual connection

        return z


class BinaryClassifier(Module):
    """A simple binary classifier for real and fake images."""

    def __init__(self, num_features: int):
        super().__init__()

        self.linear1 = Linear(num_features, num_features)
        self.linear2 = Linear(num_features, 1)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1(x)
        z = self.silu(z)
        z = self.linear2(z)

        return z
