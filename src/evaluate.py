import torch


def to_uint8_images(x):
    """Convert image tensors from [-1, 1] to uint8 [0, 255] on CPU."""
    x = ((x.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8)


@torch.no_grad()
def calculate_fid_score(
    model,
    real_loader,
    ddim_sample_fn,
    device,
    num_samples=128,
    batch_size=16,
    sampling_timesteps=50,
):
    """
    Calculate FID using real images and generated images from a DDIM sampler.

    The pretrained Inception network used by torchmetrics is used only for
    evaluation. The supplied ddim_sample_fn should generate images in [-1, 1].
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    model.eval()
    fid = FrechetInceptionDistance(feature=64, normalize=False).cpu()

    real_count = 0
    image_size = None
    channels = None

    for images, _ in real_loader:
        remaining = num_samples - real_count
        if remaining <= 0:
            break

        images = images[:remaining]
        if image_size is None:
            channels = images.shape[1]
            image_size = images.shape[-1]

        real_uint8 = to_uint8_images(images)
        fid.update(real_uint8, real=True)
        real_count += real_uint8.shape[0]

    if real_count == 0:
        raise RuntimeError("No real images were available for FID calculation.")

    fake_count = 0
    while fake_count < real_count:
        current_batch_size = min(batch_size, real_count - fake_count)
        fake_images = ddim_sample_fn(
            model=model,
            image_size=image_size,
            batch_size=current_batch_size,
            channels=channels,
            sampling_timesteps=sampling_timesteps,
            eta=0.0,
            return_intermediates=False,
        )
        fake_uint8 = to_uint8_images(fake_images)
        fid.update(fake_uint8, real=False)
        fake_count += fake_uint8.shape[0]

    return fid.compute().item()
