"""
1D U-Net for per-sample tissue segmentation.

Architecture:
  - Encoder: 5 levels of (Conv1d → BN → ReLU → Conv1d → BN → ReLU → MaxPool)
  - Bottleneck: Conv block
  - Decoder: 5 levels of (Upsample → Concat skip → Conv1d → BN → ReLU → Conv1d → BN → ReLU)
  - Output: 1x1 Conv → num_classes per sample

Input:  (batch, 1, chunk_size)      e.g. (B, 1, 4096)
Output: (batch, num_classes, chunk_size)  e.g. (B, 3, 4096)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Conv block followed by max pooling (downsampling by 2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        features = self.conv(x)
        down = self.pool(features)
        return down, features  # down goes deeper, features for skip


class DecoderBlock(nn.Module):
    """Upsample → concatenate skip → conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, kernel_size)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from pooling
        diff = skip.size(2) - x.size(2)
        if diff > 0:
            x = nn.functional.pad(x, (0, diff))
        elif diff < 0:
            skip = nn.functional.pad(skip, (0, -diff))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net for time-series segmentation.

    Parameters:
        in_channels: number of input channels (1 for raw resistance)
        num_classes: number of output classes (3: blood/clot/wall)
        base_filters: number of filters in first encoder level (doubles each level)
        depth: number of encoder/decoder levels
        kernel_size: convolution kernel size (odd number)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_filters: int = 32,
        depth: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        ch_in = in_channels
        for i in range(depth):
            ch_out = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(ch_in, ch_out, kernel_size))
            self.enc_dropouts.append(nn.Dropout1d(dropout) if dropout > 0 else nn.Identity())
            ch_in = ch_out

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch, kernel_size)
        self.bottleneck_dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        # Decoder
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = base_filters * (2 ** i)
            ch_out = skip_ch
            self.decoders.append(DecoderBlock(ch_in, skip_ch, ch_out, kernel_size))
            ch_in = ch_out

        # Output head
        self.head = nn.Conv1d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        skips = []
        for encoder, drop in zip(self.encoders, self.enc_dropouts):
            x, features = encoder(x)
            x = drop(x)
            skips.append(features)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)

        # Decoder path
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Per-sample classification
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = UNet1D(in_channels=1, num_classes=3, base_filters=32, depth=5)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 1, 4096)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 3, 4096), f"Expected (2, 3, 4096), got {y.shape}"
    print("OK — output shape matches input length.")
