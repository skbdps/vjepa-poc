"""Small dense JEPA -> actual normalized Wan VAE latent translator.

The full 1024-channel patch grid is retained as input. No coarse RGB, mask,
SAM output, target geometry, or renderer metadata enters the projector.
Coordinates denote destination grid positions, never a copied source position.
This is a bridge into a frozen generator's VAE space, not a pretrained
ControlNet and not by itself a trained video diffusion control branch.
"""
from __future__ import annotations

import torch
from torch import nn

GRID, CHANNELS, LATENT_CHANNELS, LATENT_GRID = 24, 1024, 16, 48


class _NormalizedJEPA(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(1, CHANNELS, 1, 1))
        self.register_buffer('feature_std', torch.ones(1, CHANNELS, 1, 1))
        self.register_buffer('normalization_is_fitted', torch.tensor(False))
        if (mean is None) != (std is None):
            raise ValueError('Mean and standard deviation must be supplied together')
        if mean is not None:
            self.set_normalization(mean, std)
        axis = (torch.arange(GRID, dtype=torch.float32) + .5) * (2. / GRID) - 1.
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('destination_xy', torch.stack([xx, yy])[None])

    @torch.no_grad()
    def set_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.numel() != CHANNELS or std.numel() != CHANNELS:
            raise ValueError('Expected one training mean/std per JEPA channel')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid feature normalization')
        self.feature_mean.copy_(mean.reshape(1, CHANNELS, 1, 1))
        self.feature_std.copy_(std.reshape(1, CHANNELS, 1, 1))
        self.normalization_is_fitted.fill_(True)

    def normalized_input(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or tuple(features.shape[1:]) != (CHANNELS, GRID, GRID):
            raise ValueError('Expected dense JEPA input [B,1024,24,24]')
        if not features.is_floating_point():
            raise ValueError('JEPA features must be floating point')
        if not bool(self.normalization_is_fitted):
            raise ValueError('Set training-only normalization before using the bridge')
        normalized = (features - self.feature_mean) / self.feature_std
        xy = self.destination_xy.expand(features.shape[0], -1, -1, -1)
        return torch.cat([normalized, xy], dim=1)

    def configuration(self):
        return {'architecture': type(self).__name__, 'input': [1024, 24, 24],
                'output': [16, 48, 48], 'normalization': 'training-only per-channel mean/std',
                'coordinates': 'fixed normalized destination patch centers',
                'parameters': sum(parameter.numel() for parameter in self.parameters())}


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(groups, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1))

    def forward(self, features):
        return features + self.layers(features)


class SpatialJEPAProjector(_NormalizedJEPA):
    def __init__(self, width: int = 64, mean=None, std=None):
        super().__init__(mean, std)
        if width < 8:
            raise ValueError('Width must be at least 8')
        self.width = width
        self.stem = nn.Conv2d(CHANNELS + 2, width, 1)
        self.blocks = nn.Sequential(_ResidualBlock(width), _ResidualBlock(width))
        self.readout = nn.Conv2d(width, LATENT_CHANNELS * 4, 1)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, features):
        hidden = self.stem(self.normalized_input(features))
        return self.upsample(self.readout(self.blocks(hidden)))

    def configuration(self):
        return {**super().configuration(), 'width': self.width,
                'blocks': 2, 'spatial_upsampling': '2x pixel shuffle'}


class LinearJEPAProjector(_NormalizedJEPA):
    """Shared affine per-patch comparator; no hidden nonlinear layers."""
    def __init__(self, mean=None, std=None):
        super().__init__(mean, std)
        self.readout = nn.Conv2d(CHANNELS + 2, LATENT_CHANNELS * 4, 1)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, features):
        return self.upsample(self.readout(self.normalized_input(features)))


def build_model(kind: str, mean, std, width: int = 64) -> nn.Module:
    if kind == 'cnn':
        return SpatialJEPAProjector(width=width, mean=mean, std=std)
    if kind == 'linear':
        return LinearJEPAProjector(mean=mean, std=std)
    raise ValueError('Unknown bridge architecture: ' + kind)


def as_wan_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or tuple(latents.shape[1:]) != (16, 48, 48):
        raise ValueError('Expected projected native-image latents [B,16,48,48]')
    return latents.unsqueeze(2)


def source_preserving_prediction(projector: nn.Module, source_features: torch.Tensor,
                                 edited_features: torch.Tensor,
                                 source_latents: torch.Tensor) -> torch.Tensor:
    """Analytic secondary arm: A(source) + F(edit) - F(source).

    Requires deterministic evaluation mode. Subtracting the two predictions
    before adding to source preserves an exact zero edit (no cancellation
    roundoff through the source). Returns the source tensor's 4D/5D layout.
    No claim of unchanged decoded pixels follows from this latent identity.
    """
    if projector.training:
        raise ValueError('Source-preserving evaluation requires projector.eval()')
    if source_features.shape != edited_features.shape:
        raise ValueError('Source/edit feature shapes differ')
    time_axis = source_latents.ndim == 5
    if time_axis:
        if tuple(source_latents.shape[1:]) != (16, 1, 48, 48):
            raise ValueError('Expected single-frame normalized Wan source latents')
        source = source_latents[:, :, 0]
    else:
        source = source_latents
    if source.ndim != 4 or tuple(source.shape[1:]) != (16, 48, 48):
        raise ValueError('Expected normalized Wan source latents [B,16,48,48]')
    if source.shape[0] != source_features.shape[0]:
        raise ValueError('Source latent/feature batches differ')
    delta = projector(edited_features) - projector(source_features)
    result = source + delta
    return result.unsqueeze(2) if time_axis else result
