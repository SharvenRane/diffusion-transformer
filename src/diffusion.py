import torch


class GaussianDiffusion:
    """Minimal DDPM style forward noising and a simple noise prediction loss.

    This is intentionally compact: a linear beta schedule, the closed form
    q(x_t | x_0) sampler, and the standard mean squared error on predicted
    noise. It is enough to train the DiT for a step in tests.
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_acp = self.sqrt_alphas_cumprod.to(x_start.device)[t]
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t]
        while sqrt_acp.dim() < x_start.dim():
            sqrt_acp = sqrt_acp[..., None]
            sqrt_omacp = sqrt_omacp[..., None]
        return sqrt_acp * x_start + sqrt_omacp * noise

    def training_loss(self, model, x_start: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        predicted = model(x_t, t.float(), y)
        return torch.mean((predicted - noise) ** 2)
