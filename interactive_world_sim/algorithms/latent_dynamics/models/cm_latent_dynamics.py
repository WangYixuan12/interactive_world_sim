from functools import partial
from typing import Literal, Optional

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from interactive_world_sim.algorithms.models.attention import (
    SpatialAttentionBlock,
    TemporalAttentionBlock,
)
from interactive_world_sim.algorithms.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)


class NoiseLevelSequential(nn.Sequential):
    """Sequential module that passes the noise level to each module in the sequence"""

    def forward(
        self,
        x: torch.Tensor,
        noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the NoiseLevelSequential."""
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, noise_level, external_cond)
            else:
                x = module(x)
        return x


class ResnetBlock(nn.Module):
    """ResnetBlock module."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        emb_dim: Optional[int] = None,
        cond_dim: Optional[int] = None,
        groups: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim_out, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        self.emb_layers = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim_out * 2),
            )
            if emb_dim is not None
            else None
        )

        self.cond_emb_layers = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, dim_out * 2),
            )
            if cond_dim is not None
            else None
        )

        self.skip_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the ResnetBlock."""
        h = self.in_layers(x)

        if self.emb_layers is not None:
            assert (
                emb is not None
            ), "Noise level embedding is required for this ResnetBlock"
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            emb = self.emb_layers(emb)
            emb = rearrange(emb, "b f c -> b c f 1 1")
            scale, shift = emb.chunk(2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            if cond is not None:
                assert (
                    self.cond_emb_layers is not None
                ), "Condition embedding layers are not initialized"
                cond_out = self.cond_emb_layers(cond).type(h.dtype)
                cond_out = rearrange(cond_out, "b f c -> b c f 1 1")
                cond_scale, cond_shift = cond_out.chunk(2, dim=1)
                h = h * (1 + cond_scale) + cond_shift
            h = out_rest(h)
        else:
            h = self.out_layers(h)

        return self.skip_conv(x) + h


class CMLatentDynamics(nn.Module):
    """CMLatentDynamics model."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        dim: int = 64,
        action_emb_dim: int = 128,
        resnet_block_groups: int = 8,
        dim_mults: list[int] = [1, 2, 4, 8],  # noqa
        attn_resolutions: list[int] = [1, 2, 4, 8],  # noqa
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        use_linear_attn: bool = True,
        use_init_temporal_attn: bool = True,
        init_kernel_size: int = 7,
        is_causal: bool = True,
        time_emb_type: Literal["sinusoidal", "rotary"] = "rotary",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        init_dim = dim
        out_dim = latent_dim
        channels = latent_dim

        self.channels = channels
        self.use_init_temporal_attn = use_init_temporal_attn
        self.is_causal = is_causal
        dim_mults = list(dim_mults)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:], strict=False))  # noqa
        mid_dim = dims[-1]

        noise_level_emb_dim = dim * 8
        self.noise_level_pos_embedding = nn.Sequential(
            Timesteps(dim, True, 0),
            TimestepEmbedding(
                in_channels=dim, time_embed_dim=noise_level_emb_dim // 2, dtype=dtype
            ),
        )
        self.action_emd = nn.Sequential(
            nn.Linear(action_dim, 64, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_emb_dim, dtype=dtype),
        )

        self.rotary_time_pos_embedding = (
            RotaryEmbedding(dim=attn_dim_head) if time_emb_type == "rotary" else None
        )

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
            dtype=dtype,
        )

        self.init_temporal_attn = (
            TemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=is_causal,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else nn.Identity()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # block_klass = partial(
        #     ResnetBlock, groups=resnet_block_groups, cond_dim=action_emb_dim)
        block_klass_noise = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            emb_dim=noise_level_emb_dim,
            cond_dim=action_emb_dim,
        )
        spatial_attn_klass = partial(
            SpatialAttentionBlock, heads=attn_heads, dim_head=attn_dim_head
        )
        temporal_attn_klass = partial(
            TemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=is_causal,
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            assert dim_in is not None
            assert dim_out is not None
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        NoiseLevelSequential(
                            block_klass_noise(dim_in, dim_out),
                            block_klass_noise(dim_out, dim_out),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
                                )
                                if use_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                        ),
                        # Downsample(dim_out) if not is_last else nn.Identity(),
                        nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1

        self.mid_block = NoiseLevelSequential(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            assert dim_in is not None
            assert dim_out is not None
            self.up_blocks.append(
                NoiseLevelSequential(
                    block_klass_noise(dim_out * 2, dim_in),
                    block_klass_noise(dim_in, dim_in),
                    (
                        spatial_attn_klass(
                            dim_in, use_linear=use_linear_attn and idx > 0
                        )
                        if use_attn
                        else nn.Identity()
                    ),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    # Upsample(dim_in) if not is_last else nn.Identity(),
                    nn.Identity(),
                )
            )

            curr_resolution //= 2 if not is_last else 1

        self.out = nn.Sequential(
            block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1, dtype=dtype)
        )
        self = self.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        stop_noise_levels: torch.Tensor,
        external_cond: torch.Tensor,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            x: (B,C,T,H,W)
            noise_levels: (T,B)
            stop_noise_levels: (T,B)
            external_cond: (T,B,D)
            is_causal: Optional[bool] = None

        Returns:
            torch.Tensor: (B,C,T,H,W)
        """
        noise_levels_new = rearrange(noise_levels, "f b -> b f")
        stop_noise_levels_new = rearrange(stop_noise_levels, "f b -> b f")
        noise_level_emb = self.noise_level_pos_embedding(noise_levels_new)
        stop_noise_level_emb = self.noise_level_pos_embedding(stop_noise_levels_new)
        noise_level_emb = torch.cat([noise_level_emb, stop_noise_level_emb], dim=-1)

        if external_cond is not None:
            external_cond_new = rearrange(external_cond, "t b d -> b t d")
            external_cond_new = self.action_emd(external_cond_new)
        else:
            external_cond_new = None

        x = self.init_conv(x)
        x = self.init_temporal_attn(x)
        h = x.clone()

        hs = []

        for block, downsample in self.down_blocks:
            h = block(h, noise_level_emb, external_cond_new)
            hs.append(h)
            h = downsample(h)

        h = self.mid_block(h, noise_level_emb, external_cond_new)

        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, noise_level_emb, external_cond_new)

        h = torch.cat([h, x], dim=1)
        return self.out(h)

    def get_optim_groups(self, weight_decay: float = 1e-3, lr: float = 1e-3) -> list:
        """Get optimizer groups."""
        # params_ls = []
        # for model in self.down_blocks + self.up_blocks + [self.mid_block]:
        #     params = tuple(model.parameters())
        #     params_dict = {"params": params, "lr": lr}
        #     params_ls.append(params_dict)
        # params_ls.append({"params": tuple(self.action_emd.parameters()), "lr": lr})
        # params_ls.append({"params": tuple(self.init_conv.parameters()), "lr": lr})
        # if self.use_init_temporal_attn:
        #     params_ls.append(
        #         {"params": tuple(self.init_temporal_attn.parameters()), "lr": lr}
        #     )
        # params_ls.append({"params": tuple(self.out.parameters()), "lr": lr})
        # return params_ls
        return [{"params": tuple(self.parameters()), "lr": lr}]


def test_conv2d_dyn() -> None:
    B, T_past, C, H, W = 3, 1, 4, 32, 32
    T_future = 9
    action_dim = 20
    latents = torch.randn(B, T_past, C, H, W)
    actions = torch.randn(B, T_future, action_dim)
    model = CMLatentDynamics(
        latent_dim=C,
        action_dim=action_dim,
        action_emb_dim=C,
        use_init_temporal_attn=False,
        dtype=torch.float32,
    )
    with torch.no_grad():
        preds = model(latents, actions)
    print(preds.shape)


if __name__ == "__main__":
    test_conv2d_dyn()
