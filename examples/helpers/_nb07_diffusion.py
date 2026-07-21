"""Host-side machinery for the nb07 discrete-diffusion demo.

The UNet denoiser and a Frechet feature distance. None of this touches Torx; it
only produces the per-pixel clean-bit logits the notebook then turns into Torx
``PNOT`` flip parameters. The denoiser is trained offline on full MNIST (see the
committed checkpoint under ``assets/nb07/``); the notebook only loads it.
"""

from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA


class UNet(nn.Module):
    """Two-pool U-Net denoiser (472545 params at ``base=32``).

    Definition is frozen: ``from_bytes`` only matches the offline checkpoint if
    the module structure is byte-for-byte the trained one.
    """

    base: int = 32

    @nn.compact
    def __call__(self, x):  # x: (B, 28, 28, 2) = [noisy, sigma]
        c = self.base

        def block(h, ch):
            h = nn.relu(
                nn.GroupNorm(num_groups=8)(nn.Conv(ch, (3, 3), padding="SAME")(h))
            )
            h = nn.relu(
                nn.GroupNorm(num_groups=8)(nn.Conv(ch, (3, 3), padding="SAME")(h))
            )
            return h

        s1 = block(x, c)
        d1 = nn.max_pool(s1, (2, 2), (2, 2))
        s2 = block(d1, 2 * c)
        d2 = nn.max_pool(s2, (2, 2), (2, 2))
        mid = block(d2, 4 * c)
        u2 = jax.image.resize(mid, s2.shape[:-1] + (mid.shape[-1],), "nearest")
        u2 = block(jnp.concatenate([u2, s2], -1), 2 * c)
        u1 = jax.image.resize(u2, s1.shape[:-1] + (u2.shape[-1],), "nearest")
        u1 = block(jnp.concatenate([u1, s1], -1), c)
        return nn.Conv(1, (1, 1))(u1)[..., 0]


# single module instance shared by the loader and `denoise_logits`
_MODEL = UNet()


def load_unet_params(checkpoint_path):
    """Load the offline-trained UNet params from a flax msgpack checkpoint.

    Init on a dummy ``(1, 28, 28, 2)`` only supplies the pytree structure that
    ``from_bytes`` fills with the trained weights.
    """
    init_params = _MODEL.init(
        jax.random.key(0), jnp.zeros((1, 28, 28, 2), dtype=jnp.float32)
    )
    msgpack_bytes = Path(checkpoint_path).read_bytes()
    return flax.serialization.from_bytes(init_params, msgpack_bytes)


def denoise_logits(params, noisy, sigma):
    """Per-pixel clean-bit logits for a binary batch at noise level ``sigma``.

    The denoiser sees both the noisy pixels and the scalar noise level as the
    two input channels.
    """
    noisy = jnp.asarray(noisy, dtype=jnp.float32)
    # keep sigma a tracer-safe array so eqx.filter_jit works with a dynamic level
    sigma_channel = jnp.full(noisy.shape, jnp.asarray(sigma, dtype=noisy.dtype))
    model_input = jnp.stack([noisy, sigma_channel], axis=-1)
    return _MODEL.apply(params, model_input)


def frechet_distance(feat_real: np.ndarray, feat_gen: np.ndarray) -> float:
    """Frechet distance between Gaussians fit to two feature sets.

    Same algebra as FID but over the supplied features (a PCA proxy in nb07),
    not Inception activations.
    """
    mu_r, mu_g = feat_real.mean(0), feat_gen.mean(0)
    # 1e-6 floor keeps covariances PD so `sqrtm` stays well-conditioned
    cov_r = np.cov(feat_real, rowvar=False) + 1e-6 * np.eye(feat_real.shape[1])
    cov_g = np.cov(feat_gen, rowvar=False) + 1e-6 * np.eye(feat_gen.shape[1])
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        # only roundoff-scale imaginary parts are acceptable; larger means the
        # matrix product was not PSD and the distance would be wrong
        imag_scale = np.max(np.abs(covmean.imag))
        if imag_scale > 1e-3:
            raise ValueError(f"sqrtm returned material imaginary part {imag_scale:.3e}")
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean))


def pca_features(train_images: np.ndarray, *, n_components=16, seed=0):
    """Fit a PCA on the training digits; return a (batch -> features) function."""
    flat = train_images.reshape(len(train_images), -1)
    pca = PCA(n_components=n_components, random_state=seed).fit(flat)
    return lambda b: pca.transform(b.reshape(len(b), -1))
