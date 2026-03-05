from typing import Optional

import torch as th
import torch.nn as nn

from interactive_world_sim.algorithms.models.diffae_unet import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class CMControlledUnetModel(nn.Module):
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

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1, dtype=dtype)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        self.time_embed_dim * 2,
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
                            self.time_embed_dim * 2,
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
                self.time_embed_dim * 2,
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
                self.time_embed_dim * 2,
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
                        self.time_embed_dim * 2,
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
                            self.time_embed_dim * 2,
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
            zero_module(
                conv_nd(dims, input_ch, out_channels, 3, padding=1, dtype=dtype)
            ),
        )

    def forward(  # type: ignore
        self,
        x: th.Tensor,
        t: th.Tensor,
        s: th.Tensor,
        controls: Optional[list[th.Tensor]] = None,
    ) -> th.Tensor:
        """Forward pass of the model."""
        t_emb = self.time_embed(
            timestep_embedding(t, self.model_channels, dtype=self.dtype)
        )
        s_emb = self.time_embed(
            timestep_embedding(s, self.model_channels, dtype=self.dtype)
        )
        time_emb = th.cat([t_emb, s_emb], dim=-1)

        hs = []
        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, time_emb)
            hs.append(h)
        h = self.middle_block(h, time_emb)

        if controls is not None:
            h += controls.pop()

        for module in self.output_blocks:
            if controls is None:
                h = th.cat([h, hs.pop()], dim=1)
            else:
                h = th.cat([h, hs.pop() + controls.pop()], dim=1)
            h = module(h, time_emb)
        h = h.type(x.dtype)
        o = self.out(h)

        return o


class CMControlNet(nn.Module):
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
        num_cond_upsamples: int = 0,
        num_cond_channel: Optional[int] = None,
        dtype: th.dtype = th.float32,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.dims = dims
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
        self.num_cond_upsamples = num_cond_upsamples
        self.num_cond_channel = (
            num_cond_channel if num_cond_channel is not None else cond_dim
        )

        self.cond_dim = cond_dim
        self.time_embed_dim = t_emb_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, t_emb_dim, dtype=dtype),  # type: ignore
            nn.SiLU(),
            linear(t_emb_dim, t_emb_dim, dtype=dtype),  # type: ignore
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1, dtype=dtype)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.upsample_conds = nn.ModuleList(
            [
                Upsample(self.num_cond_channel, use_conv=True, dims=dims, dtype=dtype)
                for _ in range(num_cond_upsamples)
            ]
        )
        self.input_blocks_cond = TimestepEmbedSequential(
            conv_nd(dims, self.num_cond_channel, 16, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, dtype=dtype),  # type: ignore
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1, dtype=dtype)),  # type: ignore
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        channels=ch,
                        t_emb_channels=self.time_embed_dim * 2,
                        dropout=dropout,
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
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels=ch,
                            t_emb_channels=self.time_embed_dim * 2,
                            dropout=dropout,
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
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                t_emb_channels=self.time_embed_dim * 2,
                dropout=dropout,
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
                channels=ch,
                t_emb_channels=self.time_embed_dim * 2,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels: int) -> nn.Module:
        """Make a zero-initialized convolution."""
        return TimestepEmbedSequential(
            zero_module(
                conv_nd(self.dims, channels, channels, 1, padding=0, dtype=self.dtype)
            )
        )

    def forward(
        self,
        x: th.Tensor,
        t: th.Tensor,
        s: th.Tensor,
        latent: th.Tensor,
    ) -> th.Tensor:
        """Forward pass of the model."""
        t_emb = self.time_embed(
            timestep_embedding(t, self.model_channels, dtype=self.dtype)
        )
        s_emb = self.time_embed(
            timestep_embedding(s, self.model_channels, dtype=self.dtype)
        )
        time_emb = th.cat([t_emb, s_emb], dim=-1)

        h = x.type(self.dtype)

        # compute control
        if self.num_cond_upsamples > 0:
            for upsample_cond in self.upsample_conds:
                latent = upsample_cond(latent)
        guided_hint = self.input_blocks_cond(latent, time_emb)
        controls = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs, strict=False):
            if guided_hint is not None:
                h = module(h, time_emb)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, time_emb)
            controls.append(zero_conv(h, time_emb))

        h = self.middle_block(h, time_emb)
        controls.append(self.middle_block_out(h, time_emb))
        return controls
