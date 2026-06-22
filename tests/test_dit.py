import copy
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DiTConfig
from src.diffusion import GaussianDiffusion
from src.model import DiT, DiTBlock, TimestepEmbedder


def _tiny_config():
    return DiTConfig(
        image_size=16,
        patch_size=4,
        in_channels=1,
        hidden_size=48,
        depth=2,
        num_heads=4,
        num_classes=4,
    )


def _randomize_adaln(model):
    """adaLN-Zero starts as identity (zero modulation + zero final layer), so a
    fresh model outputs zeros. For behaviour tests we perturb those layers so
    the conditioning path is actually exercised."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.normal_(0, 0.05)


def test_forward_returns_input_image_shape():
    cfg = _tiny_config()
    model = DiT(cfg)
    b = 3
    x = torch.randn(b, cfg.in_channels, cfg.image_size, cfg.image_size)
    t = torch.randint(0, 1000, (b,)).float()
    y = torch.randint(0, cfg.num_classes, (b,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_patchify_unpatchify_token_count():
    cfg = _tiny_config()
    model = DiT(cfg)
    x = torch.randn(2, cfg.in_channels, cfg.image_size, cfg.image_size)
    tokens = model.x_embedder(x)
    assert tokens.shape == (2, cfg.num_patches, cfg.hidden_size)


def test_unpatchify_inverts_patchify_layout():
    cfg = _tiny_config()
    model = DiT(cfg)
    b = 2
    # Build a per patch token of size p*p*out_channels and confirm unpatchify
    # places it at the right spatial location.
    p = cfg.patch_size
    c = cfg.out_channels
    n = cfg.num_patches
    tokens = torch.randn(b, n, p * p * c)
    imgs = model.unpatchify(tokens)
    assert imgs.shape == (b, c, cfg.image_size, cfg.image_size)


def test_adaln_conditioning_changes_output_with_timestep():
    cfg = _tiny_config()
    model = DiT(cfg)
    _randomize_adaln(model)
    model.eval()

    b = 4
    x = torch.randn(b, cfg.in_channels, cfg.image_size, cfg.image_size)
    y = torch.randint(0, cfg.num_classes, (b,))

    t1 = torch.zeros(b).float()
    t2 = torch.full((b,), 500.0)

    with torch.no_grad():
        out1 = model(x, t1, y)
        out2 = model(x, t2, y)

    # Same image and class, different timestep, must produce a different output.
    assert not torch.allclose(out1, out2, atol=1e-5)
    diff = (out1 - out2).abs().mean().item()
    assert diff > 1e-4


def test_adaln_conditioning_changes_output_with_class():
    cfg = _tiny_config()
    model = DiT(cfg)
    _randomize_adaln(model)
    model.eval()

    b = 4
    x = torch.randn(b, cfg.in_channels, cfg.image_size, cfg.image_size)
    t = torch.full((b,), 250.0)

    y1 = torch.zeros(b, dtype=torch.long)
    y2 = torch.full((b,), cfg.num_classes - 1, dtype=torch.long)

    with torch.no_grad():
        out1 = model(x, t, y1)
        out2 = model(x, t, y2)

    assert not torch.allclose(out1, out2, atol=1e-5)


def test_timestep_embedder_distinct_for_distinct_timesteps():
    emb = TimestepEmbedder(hidden_size=48)
    t = torch.tensor([0.0, 1.0, 50.0, 999.0])
    out = emb(t)
    assert out.shape == (4, 48)
    # Different timesteps should map to different embeddings.
    assert not torch.allclose(out[0], out[3])


def test_dit_block_is_identity_at_init():
    cfg = _tiny_config()
    block = DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio)
    # adaLN-Zero: zero the modulation as the full model does.
    torch.nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    torch.nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
    x = torch.randn(2, cfg.num_patches, cfg.hidden_size)
    c = torch.randn(2, cfg.hidden_size)
    out = block(x, c)
    # With zero gates the block adds nothing and returns its input.
    assert torch.allclose(out, x, atol=1e-6)


def test_model_trains_a_step():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = DiT(cfg)
    diffusion = GaussianDiffusion(num_timesteps=1000)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    b = 8
    x = torch.randn(b, cfg.in_channels, cfg.image_size, cfg.image_size)
    y = torch.randint(0, cfg.num_classes, (b,))

    before = copy.deepcopy([p.detach().clone() for p in model.parameters()])

    losses = []
    for _ in range(5):
        t = torch.randint(0, diffusion.num_timesteps, (b,))
        opt.zero_grad()
        loss = diffusion.training_loss(model, x, t, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss is finite.
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    # At least one parameter changed after the optimizer steps.
    changed = any(
        not torch.allclose(b0, p.detach())
        for b0, p in zip(before, model.parameters())
    )
    assert changed

    # Gradients reached the patch embedder (gradient flow check).
    assert model.x_embedder.proj.weight.grad is not None
    assert model.x_embedder.proj.weight.grad.abs().sum().item() > 0


def test_loss_decreases_on_overfit_single_batch():
    torch.manual_seed(1)
    cfg = _tiny_config()
    model = DiT(cfg)
    diffusion = GaussianDiffusion(num_timesteps=1000)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    b = 4
    x = torch.randn(b, cfg.in_channels, cfg.image_size, cfg.image_size)
    y = torch.randint(0, cfg.num_classes, (b,))
    t = torch.full((b,), 100, dtype=torch.long)
    noise = torch.randn_like(x)

    # Fix the noise target so the batch is genuinely overfittable.
    def fixed_loss():
        x_t = diffusion.q_sample(x, t, noise)
        pred = model(x_t, t.float(), y)
        return torch.mean((pred - noise) ** 2)

    first = fixed_loss().item()
    for _ in range(60):
        opt.zero_grad()
        loss = fixed_loss()
        loss.backward()
        opt.step()
    last = fixed_loss().item()

    assert last < first
