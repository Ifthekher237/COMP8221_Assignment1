import torch


def make_beta_schedule(timesteps, beta_start, beta_end, device):
    """Create a linear beta schedule for the forward diffusion process."""
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def prepare_diffusion_schedule(config, device):
    """Prepare reusable diffusion schedule tensors on the selected device."""
    betas = make_beta_schedule(
        timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        device=device,
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([
        torch.ones(1, device=device),
        alphas_cumprod[:-1],
    ])

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    }


def extract(a, t, x_shape):
    """Gather timestep coefficients and reshape them for image broadcasting."""
    batch_size = t.shape[0]
    t = t.to(device=a.device, dtype=torch.long)
    values = a.gather(0, t)
    return values.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(x_start, t, schedule, noise=None):
    """
    Apply the forward diffusion process q(x_t | x_0).

    The input images are expected to be scaled to [-1, 1]. The returned tensor
    has the same shape as x_start and contains the noisy image x_t.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_bar_t = extract(schedule["sqrt_alphas_cumprod"], t, x_start.shape)
    sqrt_one_minus_alpha_bar_t = extract(
        schedule["sqrt_one_minus_alphas_cumprod"],
        t,
        x_start.shape,
    )
    return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise


@torch.no_grad()
def ddim_sample(
    model,
    schedule,
    image_size,
    batch_size,
    channels,
    device,
    sampling_timesteps,
    eta=0.0,
    return_intermediates=False,
):
    """
    Generate images with DDIM reverse sampling from pure Gaussian noise.

    DDIM uses a reduced sequence of reverse timesteps and can be deterministic
    when eta is 0. The function returns images in the model's native [-1, 1]
    scale and does not perform plotting or saving.
    """
    model.eval()
    training_timesteps = schedule["alphas_cumprod"].shape[0]
    img = torch.randn(batch_size, channels, image_size, image_size, device=device)
    step_sequence = torch.linspace(
        training_timesteps - 1,
        0,
        sampling_timesteps,
        device=device,
    ).long()

    def get_alpha_bar(timestep):
        if int(timestep.item()) < 0:
            return torch.ones((), device=device)
        return schedule["alphas_cumprod"][timestep]

    intermediates = []
    if return_intermediates:
        intermediates.append(img.detach().clone())

    for step_idx, timestep in enumerate(step_sequence):
        if step_idx < len(step_sequence) - 1:
            next_timestep = step_sequence[step_idx + 1]
        else:
            next_timestep = torch.tensor(-1, device=device, dtype=torch.long)

        t_batch = torch.full(
            (batch_size,),
            int(timestep.item()),
            device=device,
            dtype=torch.long,
        )
        predicted_noise = model(img, t_batch)

        alpha_bar_t = get_alpha_bar(timestep).reshape(1, 1, 1, 1)
        alpha_bar_next = get_alpha_bar(next_timestep).reshape(1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))

        pred_x0 = (img - sqrt_one_minus_alpha_bar_t * predicted_noise) / torch.clamp(
            sqrt_alpha_bar_t,
            min=1e-8,
        )
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if eta == 0.0:
            img = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(
                torch.clamp(1.0 - alpha_bar_next, min=0.0)
            ) * predicted_noise
        else:
            sigma = eta * torch.sqrt(torch.clamp(
                ((1.0 - alpha_bar_next) / torch.clamp(1.0 - alpha_bar_t, min=1e-8))
                * (1.0 - alpha_bar_t / torch.clamp(alpha_bar_next, min=1e-8)),
                min=0.0,
            ))
            direction = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_next - sigma ** 2, min=0.0)
            ) * predicted_noise
            img = torch.sqrt(alpha_bar_next) * pred_x0 + direction + sigma * torch.randn_like(img)

        if return_intermediates:
            intermediates.append(img.detach().clone())

    if return_intermediates:
        return img, intermediates
    return img
