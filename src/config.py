from dataclasses import dataclass


@dataclass
class DiTConfig:
    """Configuration for a tiny Diffusion Transformer.

    Defaults are deliberately small so the model runs on CPU in tests.
    """

    image_size: int = 16
    patch_size: int = 4
    in_channels: int = 1
    hidden_size: int = 48
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 4
    learn_sigma: bool = False

    @property
    def num_patches_per_side(self) -> int:
        assert self.image_size % self.patch_size == 0, (
            "image_size must be divisible by patch_size"
        )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side ** 2

    @property
    def out_channels(self) -> int:
        return self.in_channels * 2 if self.learn_sigma else self.in_channels
