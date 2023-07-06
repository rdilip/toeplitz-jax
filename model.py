import equinox as eqx
import equinox.nn as nn
import jax.nn as jnn
from einops import rearrange, reduce


class GEGLU(eqx.Module):
    """From https://arxiv.org/pdf/2002.05202.pdf%5C%5D"""

    def __init__(self, dim: int):
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc3 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x):
        h1 = self.fc1(x)
        weight = jnn.gelu(h1)
        h2 = self.fc2(x)
        h = weight * h2
        h = self.fc3(h)
        return jnn.gelu(h)


class RPE(eqx.Module):
    # starting from integer input, you can also project to sine and cosine functions to encode integers
    def __init__(self, latent_dim: int, output_dim: int, num_layers: int):

        self.projection = nn.Linear(1, self.latent_dim, bias=True)

        self.latent_operators = [
            nn.Sequential(
                nn.LayerNorm(),
                nn.Lambda(jnn.gelu),
                nn.Linear(latent_dim, latent_dim, bias=False),
            )
            for _ in range(num_layers)
        ]

        self.output_operaotr = nn.Sequential(
            nn.LayerNorm(),
            nn.Lambda(jnn.gelu),
            nn.Linear(latent_dim, output_dim, bias=False),
        )
    
    def __call__(self, x):
        x = self.projection(x)
        for op in self.latent_operators:
            x = op(x) + x
        x = self.output_operaotr(x)
        return x
