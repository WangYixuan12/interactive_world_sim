import math
from abc import abstractmethod
from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        swish: float,
        eps: float = 1e-5,
        dtype: th.dtype = th.float32,
    ) -> None:
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, dtype=dtype
        )
        self.swish = swish

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass of the module."""
        y = super().forward(x).to(x.dtype)
        # y = super().forward(x).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


def conv_nd(dims: int, *args: list, **kwargs: dict) -> nn.Module:
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args: list, **kwargs: dict) -> nn.Module:
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args: list, **kwargs: dict) -> nn.Module:
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def normalization(
    channels: int,
    swish: float = 0.0,
    num_groups: int = 16,
    dtype: th.dtype = th.float32,
) -> GroupNorm32:
    """Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(
        num_channels=channels, num_groups=num_groups, swish=swish, dtype=dtype
    )  # used to be 32


def timestep_embedding(
    timesteps: th.Tensor,
    dim: int,
    max_period: int = 10000,
    dtype: th.dtype = th.float32,
) -> th.Tensor:
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=dtype) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(dtype) * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def update_ema(
    target_params: list[nn.Parameter],
    source_params: list[nn.Parameter],
    rate: float = 0.99,
) -> None:
    """Update EMA

    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params, strict=False):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def mean_flat(tensor: th.Tensor) -> th.Tensor:
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Time embedding sequential module.

    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(  # type: ignore
        self,
        x: th.Tensor,
        emb: th.Tensor,
        cond: Optional[th.Tensor] = None,
        encoder_out: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Forward pass of the module."""
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, cond)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        dtype: th.dtype = th.float32,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1, dtype=dtype)  # type: ignore

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass of the module."""
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        dtype: th.dtype = th.float32,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1, dtype=dtype  # type: ignore
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)  # type: ignore

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass of the module."""
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        t_emb_channels: int,
        cond_channels: Optional[int] = None,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        dtype: th.dtype = th.float32,
    ):
        super().__init__()
        self.channels = channels
        self.t_emb_channels = t_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0, dtype=dtype),
            nn.Identity(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=dtype),  # type: ignore
            # normalization(self.out_channels, swish=1.0),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(t_emb_channels, out_channels, dtype=dtype),  # type: ignore
            # nn.LayerNorm(out_channels),
        )
        if cond_channels is not None:
            self.cond_emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(cond_channels, out_channels, dtype=dtype),  # type: ignore
                # nn.LayerNorm(out_channels),
            )
        self.out_layers = nn.Sequential(
            normalization(
                self.out_channels,
                swish=0.0 if use_scale_shift_norm else 1.0,
                dtype=dtype,
            ),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, dtype=dtype)  # type: ignore
            ),
            # normalization(self.out_channels,swish=0.0 if use_scale_shift_norm else 1.0),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, dtype=dtype  # type: ignore
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, dtype=dtype)  # type: ignore

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

    def forward(
        self, x: th.Tensor, emb: th.Tensor, cond: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        if emb_out.ndim == 3:
            emb_out = rearrange(emb_out, "b t d -> b d t 1 1")
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if cond is not None:
            cond_out = self.cond_emb_layers(cond).type(h.dtype)
            while len(cond_out.shape) < len(h.shape):
                cond_out = cond_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            if cond is not None:
                cond_scale, cond_shift = th.chunk(cond_out, 2, dim=1)
                h = h * (1 + cond_scale) + cond_shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        encoder_channels: Optional[int] = None,
        dtype: th.dtype = th.float32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by \
                    num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, swish=0.0, dtype=dtype)
        self.qkv = conv_nd(1, channels, channels * 3, 1, dtype=dtype)  # type: ignore
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1, dtype=dtype)  # type: ignore
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1, dtype=dtype))  # type: ignore

    def forward(
        self, x: th.Tensor, encoder_out: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """Forward pass of the module."""
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """A module which performs QKV attention.

    Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(
        self, qkv: th.Tensor, encoder_kv: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class DiffAEUNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list,
        t_emb_dim: int = 256,
        cond_dim: int = 256,
        dropout: float = 0,
        channel_mult: tuple = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        encoder_channels: Optional[int] = None,
        num_components: int = 1,
        dtype: th.dtype = th.float32,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.cond_dim = cond_dim
        self.time_embed_dim = t_emb_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, t_emb_dim, dtype=dtype),  # type: ignore
            nn.SiLU(),
            linear(t_emb_dim, t_emb_dim, dtype=dtype),  # type: ignore
        )

        self.num_components = num_components

        self.latent_dim = (
            self.time_embed_dim
        )  # default set latent_dim and time_embed_dim to be the same

        self.latent_dim_expand = self.latent_dim * self.num_components

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1, dtype=dtype))]  # type: ignore
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.cond_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                            dtype=dtype,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=dtype,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=dtype,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                dtype=dtype,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.cond_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                            dtype=dtype,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=dtype,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=dtype,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, swish=1.0, dtype=dtype),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, dtype=dtype)),  # type: ignore
        )

    def forward(
        self,
        x: th.Tensor,
        t: th.Tensor,
        latent: th.Tensor,
        latent_index: Optional[int] = None,
    ) -> th.Tensor:
        """Forward pass of the model."""
        time_emb = self.time_embed(
            timestep_embedding(t, self.model_channels, dtype=self.dtype)
        )

        s = x.size()
        b = s[0]
        if latent_index is None:
            # duplicate latent, time_emb, and x enough times
            latent = latent.view(b, self.num_components, self.cond_dim)
            time_emb = time_emb[:, None, :].expand(-1, self.num_components, -1)

            x = x[:, None, :].expand(-1, self.num_components, -1, -1, -1)
            x = th.flatten(x, 0, 1)

            time_emb = th.flatten(time_emb, 0, 1)
            latent = th.flatten(latent, 0, 1)

        else:
            # take given slice
            latent = latent[
                :, self.latent_dim * latent_index : self.latent_dim * (latent_index + 1)
            ]

        hs = []
        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, time_emb, latent)
            hs.append(h)
        h = self.middle_block(h, time_emb, latent)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, time_emb, latent)
        h = h.type(x.dtype)
        o = self.out(h)

        s = o.size()
        if latent_index is None:
            o = o.view(b, -1, *s[1:]).mean(dim=1)
        return o
