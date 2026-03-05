import os
import tracemalloc
from typing import Any, Callable

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

# import matplotlib.pyplot as plt
from interactive_world_sim.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.common.metrics import (
    FrechetInceptionDistance,
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from interactive_world_sim.algorithms.models.cm_decoder import CMDecoder
from interactive_world_sim.algorithms.models.utils import EinopsWrapper
from interactive_world_sim.utils.cm_utils import DDPMScheduler
from interactive_world_sim.utils.logging_utils import (
    get_validation_metrics_for_videos,
    log_video,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


class LatentWorldModel(BasePytorchAlgo):
    """StudentV1_0"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metrics = cfg.metrics
        self.num_latent_channel = cfg.num_latent_channel
        self.num_latent_downsample = cfg.num_latent_downsample
        self.training_stage = cfg.training_stage
        assert self.training_stage in [1, 2, 3], "Invalid training stage"
        self.load_ae = cfg.load_ae if "load_ae" in cfg else None
        super().__init__(cfg)
        self.normalizer = LinearNormalizer()
        self.validation_step_outputs: list = []
        self.validation_metrics: dict = {}
        self.timesteps: int = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.obs_keys = cfg.obs_keys
        self.val_render = cfg.val_render
        self.clip_noise = self.cfg.diffusion.clip_noise
        self.guidance_scale = self.cfg.guidance_scale
        self.n_tokens = self.cfg.n_frames
        self.mask_prev_action = (
            cfg.mask_prev_action if "mask_prev_action" in cfg else False
        )
        self.num_views = len(self.obs_keys)

        self.latent_resolution = cfg.latent_resolution
        self.noise_scheduler: DDPMScheduler = hydra.utils.instantiate(
            cfg.noise_scheduler
        )

        self.debug = False
        self.lr_scheduler = cfg.lr_scheduler if "lr_scheduler" in cfg else "linear"
        self.sampling_strategy = (
            cfg.sampling_strategy if "sampling_strategy" in cfg else "uniform"
        )
        self.prev_frame_noise_scale = (
            cfg.prev_frame_noise_scale if "prev_frame_noise_scale" in cfg else 0.1
        )
        self.dyn_infer_steps = cfg.dyn_infer_steps if "dyn_infer_steps" in cfg else 1
        self.dec_infer_steps = cfg.dec_infer_steps if "dec_infer_steps" in cfg else 1
        self.last_frame_loss_only = (
            cfg.last_frame_loss_only if "last_frame_loss_only" in cfg else False
        )
        self.robust_latent = cfg.robust_latent if "robust_latent" in cfg else False

    def _build_model(self) -> None:
        # decoder
        self.decoder: CMDecoder = CMDecoder(
            self.cfg.x_shape,
            self.cfg.latent_dim,
            self.cfg.diffusion,
            dtype=self.dtype,
        )

        # dynamics
        self.dynamics: EinopsWrapper = EinopsWrapper(
            from_shape="f b c h w",
            to_shape="b c f h w",
            module=hydra.utils.instantiate(self.cfg.dynamics),
        )

        # encoder
        latent_ch = self.num_latent_channel
        encoder_module_ls = [nn.Conv2d(self.cfg.x_shape[0], latent_ch, 3, padding=1)]
        for _ in range(self.num_latent_downsample):
            encoder_module_ls.extend(
                [
                    nn.SiLU(),
                    nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, stride=2),
                ]
            )
        self.encoder = nn.Sequential(*encoder_module_ls)

        # load previous trained model
        if self.load_ae is not None:
            cfg_cp = self.cfg.copy()
            load_ae_dir = os.path.dirname(os.path.dirname(self.load_ae))
            cfg_path = f"{load_ae_dir}/.hydra/config.yaml"
            cfg_cp = OmegaConf.load(cfg_path)
            cfg_cp.load_ae = None
            diffae = LatentWorldModel.load_from_checkpoint(
                self.load_ae,
                cfg=cfg_cp.algorithm,
                map_location=self.device,
                weights_only=False,
            )
            self.encoder.load_state_dict(diffae.encoder.state_dict())
            if self.training_stage == 3:
                self.dynamics.load_state_dict(diffae.dynamics.state_dict())
            self.decoder.load_state_dict(diffae.decoder.state_dict())

        self.validation_fid_model = (
            FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        )
        self.validation_lpips_model = (
            LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        )
        self.validation_fvd_model: FrechetVideoDistance = (
            FrechetVideoDistance() if "fvd" in self.metrics else None
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set the normalizer for the model"""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the model"""
        if self.training_stage == 1:
            param_groups = [
                {"params": self.decoder.parameters(), "lr": self.cfg.lr},
                {"params": self.encoder.parameters(), "lr": self.cfg.lr},
            ]
        elif self.training_stage == 2:
            param_groups = [
                {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            ]
        elif self.training_stage == 3:
            param_groups = [
                {"params": self.decoder.parameters(), "lr": self.cfg.lr * 0.1},
            ]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        if self.lr_scheduler == "linear":
            lr_scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=self.cfg.warmup_steps,
            )
        elif self.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=50000,
                verbose=True,
                threshold=1e-3,
                threshold_mode="rel",
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_scheduler} not included")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "training/loss",
                "strict": True,
                "name": "lr_scheduler",
            },
        }

    def encoder_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder

        Args:
            obs: (B, C, H, W)

        Returns:
            z: (B, C_latent, H_latent, W_latent)
        """
        assert (
            len(obs.shape) == 4
        ), f"Expected obs to have shape (B, C, H, W) but got {obs.shape}"
        z = self.encoder(obs)
        num_views = len(self.obs_keys)
        c_per_v = z.shape[1] // num_views
        for i in range(num_views):
            z_chunk = z[:, i * c_per_v : (i + 1) * c_per_v].clone()
            z[:, i * c_per_v : (i + 1) * c_per_v] = z_chunk / (
                torch.norm(z_chunk, dim=(1), keepdim=True) + 1e-8
            )
        return z

    def optimizer_step(
        self,
        epoch: dict,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Callable,
    ) -> None:
        """Override the optimizer step to manually warm up the learning rate"""
        # update params
        optimizer.step(closure=optimizer_closure)
        if self.training_stage == 2:
            for name, param in self.dynamics.named_parameters():
                if (
                    param.requires_grad
                    and (param.grad is not None)
                    and torch.isnan(param.grad).any()
                ):
                    print(f"NaN in gradient of {name}")
                    print(f"Parameter: {name}, Gradient: {param.grad}")
                    print(f"Parameter: {name}, Value: {param.data}")
                    print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
                    print(f"Parameter: {name}, Shape: {param.shape}")
                    print(f"Parameter: {name}, Device: {param.device}")
                    print(f"Parameter: {name}, Type: {param.dtype}")
                    print(f"Parameter: {name}, Isnan: {torch.isnan(param).any()}")
                    print(f"Parameter: {name}, Isinf: {torch.isinf(param).any()}")
                    exit()

    # ========= forward  ============
    def _forward(
        self,
        model: Any,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        stop_time: torch.Tensor,
        external_cond: Any = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the model"""
        assert (timestep >= stop_time).all()
        assert (timestep[-1] > stop_time[-1]).all()
        denoise = lambda x, t, s: model(x, t, s, external_cond=external_cond)
        return self.noise_scheduler.CTM_calc_out(
            denoise, sample, timestep, stop_time, clamp=clamp
        )

    # ========= inference  ============
    @torch.no_grad()
    def dynamics_forward(self, z_0: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """dynamics forward pass"""
        z_0 = rearrange(z_0, "b t c h w -> t b c h w")  # (T_hist, B, C, H, W)
        action = rearrange(action, "b t c -> t b c")  # (T_hist + T_act, B, A)
        T_hist = z_0.shape[0]
        T_act = action.shape[0] - T_hist
        chunk_size = 1
        curr_end = T_hist + chunk_size
        total_frames = T_hist + T_act
        xs_pred = z_0.clone()
        batch_size = z_0.shape[1]

        # pbar = tqdm(total=total_frames, initial=curr_end, desc="Sampling")
        while curr_end <= total_frames:
            horizon = chunk_size

            chunk = torch.randn(
                (horizon, batch_size, *z_0.shape[2:]),
                device=self.device,
                dtype=self.dtype,
            )
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            curr_start = max(0, curr_end - self.n_tokens)

            # pbar.set_postfix(
            #     {
            #         "start": curr_start,
            #         "end": curr_end,
            #     }
            # )

            clean_t = (
                torch.ones((xs_pred[curr_start:].shape[0] - 1,), device=self.device)
                * self.noise_scheduler.stabilization_level
            )
            timesteps = torch.linspace(
                self.noise_scheduler.timesteps - 1,
                0,
                self.dyn_infer_steps + 1,
                device=z_0.device,
            )
            action_chunk = action[curr_start:curr_end]
            if self.mask_prev_action:
                action_chunk[:-1] = 0

            for step_i in range(self.dyn_infer_steps):
                t = timesteps[step_i].unsqueeze(0)
                s = timesteps[step_i + 1].unsqueeze(0)
                t = torch.cat([clean_t, t], 0)
                t = torch.tile(t[:, None], (1, xs_pred.shape[1]))
                s = torch.cat([clean_t, s], 0)
                s = torch.tile(s[:, None], (1, xs_pred.shape[1]))
                t = t.long()
                s = s.long()
                xs_pred_updated = self._forward(
                    self.dynamics,
                    xs_pred[curr_start:],
                    t,
                    s,
                    external_cond=action_chunk,
                )  # clamp at inference time
                if self.last_frame_loss_only:
                    xs_pred[-1:] = xs_pred_updated[-1:]
                else:
                    xs_pred[curr_start:] = xs_pred_updated

            curr_end += horizon
            # pbar.update(horizon)

        # normalization
        num_views = len(self.obs_keys)
        c_per_v = xs_pred.shape[2] // num_views
        for i in range(num_views):
            xs_pred_chunk = xs_pred[:, :, i * c_per_v : (i + 1) * c_per_v].clone()
            xs_pred[:, :, i * c_per_v : (i + 1) * c_per_v] = xs_pred_chunk / (
                torch.norm(xs_pred_chunk, dim=(2), keepdim=True) + 1e-8
            )
        xs_pred = rearrange(xs_pred[T_hist:], "t b c h w -> b t c h w")
        return xs_pred

    def validation_step(
        self, batch: dict, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        """Validation step of the model"""
        # compute diffusion loss
        # (B, T, C, H, W)
        obs_ls = [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys]
        obs = torch.cat(obs_ls, dim=2)
        action = self.normalizer["action"].normalize(batch["action"])  # (B, T, A)

        obs = obs.float()
        action = action.float()

        # compute gt latent
        xs = obs
        xs = rearrange(xs, "b t c h w -> (b t) c h w")
        z_gt = self.encoder_forward(xs)
        z_gt = rearrange(z_gt, "(b t) c h w -> b t c h w", b=obs.shape[0])

        if self.training_stage in [1]:
            # compute predicted latent
            z_seq = z_gt
        elif self.training_stage in [2]:
            # compute predicted latent
            z_0 = z_gt[:, 0]
            z_seq_ls = []
            z_last = z_0.clone()
            horizon = z_gt.shape[1]

            for i in range(1, action.shape[1], horizon):
                action_chunk = action[:, i : i + horizon]  # (B, horizon, A)
                init_action_size = action_chunk.shape[1]
                if init_action_size < horizon:
                    # pad the last action to match the horizon
                    action_chunk = F.pad(
                        action_chunk,
                        (0, 0, 0, horizon - action_chunk.shape[1]),
                        mode="replicate",
                    )
                z_seq = self.dynamics_forward(
                    z_last[:, None],
                    action_chunk,
                )  # (B, T, latent_dim)
                z_seq = z_seq[:, :init_action_size]
                z_seq_ls.append(z_seq)
                z_last = z_seq[:, -1].clone()
            z_seq = torch.cat(z_seq_ls, 1)
            z_seq = torch.cat([z_0.unsqueeze(1), z_seq], 1)  # (B, T, latent_dim)
            val_loss = F.mse_loss(z_seq, z_gt, reduction="none")  # (B, T, latent_dim)
            if torch.isnan(val_loss).any():
                print("NaN in val_loss")
            val_loss = val_loss[:, 1:].mean()
            self.log(f"{namespace}/dyn_loss", val_loss)
            if "dyn_loss" not in self.validation_metrics:
                self.validation_metrics["dyn_loss"] = []
            self.validation_metrics["dyn_loss"].append(val_loss)
        else:
            z_seq = z_gt
        z_seq = rearrange(z_seq, "b t c h w -> (b t) c h w")

        # render images
        if self.val_render:
            xs_pred = render_img_cm(
                self, z_seq, xs.shape[-1], self.normalizer, num_views=self.num_views
            )
            xs_pred = rearrange(xs_pred, "(b t) c h w -> t b c h w", b=obs.shape[0])
            xs = torch.cat([batch["obs"][k] for k in self.obs_keys], dim=2)
            xs = rearrange(xs, "b t c h w -> t b c h w", b=obs.shape[0])
            xs_pred = xs_pred.detach().cpu()
            xs = xs.detach().cpu()
            self.validation_step_outputs.append((xs_pred, xs))
        return

    # ========= training  ============
    def _generate_ctm_noise_levels(self, xs: torch.Tensor) -> torch.Tensor:
        """Generate noise levels for training."""
        num_frames, batch_size, *_ = xs.shape
        min_t = self.noise_scheduler.stabilization_level + 1
        last_t = torch.randint(100 + min_t + 2, self.timesteps, (batch_size,))
        last_s = torch.cat(
            [torch.randint(min_t, int(t_i.item()) - 1, (1,)) for t_i in last_t]
        )
        last_u_ls = []
        for s_i, t_i in zip(last_s, last_t, strict=False):
            min_u = max(s_i.item() + 1, int(t_i.item() - 100))
            last_u_ls.append(torch.randint(min_u, int(t_i.item()), (1,)))
        last_u = torch.cat(last_u_ls)
        last_t = last_t.unsqueeze(0).to(xs.device)
        last_s = last_s.unsqueeze(0).to(xs.device)
        last_u = last_u.unsqueeze(0).to(xs.device)

        prev_noise_levels = torch.randint(
            min_t,
            int(self.timesteps * 0.1),
            (num_frames - 1, batch_size),
            device=xs.device,
        )
        t = torch.cat([prev_noise_levels, last_t], 0)
        s = torch.cat([prev_noise_levels, last_s], 0)
        u = torch.cat([prev_noise_levels, last_u], 0)

        return t, s, u

    def _generate_noise_levels(
        self, xs: torch.Tensor, cm_steps: int = -1
    ) -> torch.Tensor:
        """Generate noise levels for training."""
        num_frames, batch_size, *_ = xs.shape

        if self.sampling_strategy == "uniform":
            last_t = torch.randint(2, self.timesteps, (batch_size,))
            last_s = torch.cat(
                [torch.randint(1, int(t_i.item()), (1,)) for t_i in last_t]
            )
            last_t = last_t.unsqueeze(0).to(xs.device)
            last_s = last_s.unsqueeze(0).to(xs.device)
        elif self.sampling_strategy == "terminal_only":
            last_t = torch.ones((batch_size,)) * (self.timesteps - 1)
            last_t = last_t.unsqueeze(0).to(xs.device)
            if cm_steps == 1:
                last_s = torch.zeros((batch_size,))
                last_s = last_s.unsqueeze(0).to(xs.device)
            else:
                intermediate_s = np.linspace(
                    0, self.timesteps - 1, cm_steps + 1, dtype=int
                )
                s_val = np.random.choice(intermediate_s[1:-1], size=(batch_size,))
                last_s = torch.ones((batch_size,)) * s_val
                last_s = last_s.unsqueeze(0).to(xs.device)

        prev_noise_levels = torch.randint(
            1,
            int(self.timesteps * self.prev_frame_noise_scale),
            (num_frames - 1, batch_size),
            device=xs.device,
        )
        t = torch.cat([prev_noise_levels, last_t], 0)
        s = torch.cat([prev_noise_levels, last_s], 0)

        return t.long(), s.long()

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        """Training step of the model"""
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None
        # normalize input
        if batch_idx % 1000 == 0:
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.compare_to(self.tracemalloc_snapshot, "lineno")

            print(f"\n[ Top 10 memory diff from start to step {batch_idx} ]")
            for stat in top_stats[:10]:
                print(stat)
        assert "valid_mask" not in batch
        obs_ls = [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys]
        obs = torch.cat(obs_ls, dim=2)
        action = self.normalizer["action"].normalize(batch["action"])  # (B, T, A)

        obs = obs.float()
        action = action.float()

        xs = obs  # (B, T, C, H, W)
        xs = rearrange(xs, "b t c h w -> (b t) c h w")

        output_dict = {}

        # generate impainting mask

        if self.training_stage == 1:
            # stage 1: train encoder and decoder
            z = self.encoder_forward(xs)  # (B*T, C, H, W)
            if self.robust_latent:
                z += torch.randn_like(z) * 0.02

            t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)  # (1, B)
            weights_t = self.noise_scheduler.get_weights(t)[0]  # (1, B)
            weights_s = self.noise_scheduler.get_weights(s)[0]  # (1, B)
            noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(
                xs[None], t, s
            )  # (1, B, C, H, W)
            noisy_xs_t = noisy_xs_t.squeeze(0)  # (B, C, H, W)
            noisy_xs_s = noisy_xs_s.squeeze(0)  # (B, C, H, W)
            t = t.squeeze(0)  # (B)
            s = s.squeeze(0)  # (B)

            u = torch.zeros_like(t).to(self.device)
            pred_s = self._forward(
                self.decoder,
                noisy_xs_t,
                t,
                s,
                external_cond=z,
            )
            if self.dec_infer_steps > 1:
                pred_u = self._forward(
                    self.decoder,
                    noisy_xs_s,
                    s,
                    u,
                    external_cond=z,
                )

            if self.last_frame_loss_only:
                loss_s = F.mse_loss(
                    pred_s[-1:], noisy_xs_s[-1:].detach(), reduction="none"
                )
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )[-1:]
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u[-1:], xs[-1:].detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_u.ndim - 2))
                    )[-1:]
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()
            else:
                loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 1))
                )
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_s.ndim - 1))
                    )
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()

            self.log("training/rec_loss", loss)
            output_dict = {
                "loss": loss,
            }
            return output_dict
        elif self.training_stage == 2:
            # stage 2: train dynamics
            with torch.no_grad():
                z = self.encoder_forward(xs)  # (B*T, C, H, W)
            z = rearrange(z, "(b t) c h w -> t b c h w", b=obs.shape[0])
            action = rearrange(action, "b t a -> t b a")

            t, s = self._generate_noise_levels(z, self.dyn_infer_steps)
            weights_t = self.noise_scheduler.get_weights(t)
            weights_s = self.noise_scheduler.get_weights(s)
            noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t, s)

            u = torch.zeros_like(t).to(self.device)
            if self.mask_prev_action:
                action[:-1] = 0
            pred_s = self._forward(
                self.dynamics,
                noisy_z_t,
                t,
                s,
                external_cond=action,
            )
            if self.dyn_infer_steps > 1:
                pred_u = self._forward(
                    self.dynamics,
                    noisy_z_s,
                    s,
                    u,
                    external_cond=action,
                )

            if self.last_frame_loss_only:
                loss_s = F.mse_loss(
                    pred_s[-1:], noisy_z_s[-1:].detach(), reduction="none"
                )
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )[-1:]
                loss_s = loss_s * weights_t
                if self.dyn_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u[-1:], z[-1:].detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_u.ndim - 2))
                    )[-1:]
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()
            else:
                loss_s = F.mse_loss(pred_s, noisy_z_s.detach(), reduction="none")
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )
                loss_s = loss_s * weights_t
                if self.dyn_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u, z.detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_s.ndim - 2))
                    )
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()

            output_dict["loss"] = loss

            self.log("training/loss", output_dict["loss"])
            for key in output_dict.keys():
                self.log(f"training/{key}", output_dict[key])
        elif self.training_stage == 3:
            with torch.no_grad():
                z = self.encoder_forward(xs)  # (B*T, C, H, W)
                z += torch.randn_like(z) * 0.02

            t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)  # (1, B)
            weights_t = self.noise_scheduler.get_weights(t)[0]  # (1, B)
            weights_s = self.noise_scheduler.get_weights(s)[0]  # (1, B)
            noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(
                xs[None], t, s
            )  # (1, B, C, H, W)
            noisy_xs_t = noisy_xs_t.squeeze(0)  # (B, C, H, W)
            noisy_xs_s = noisy_xs_s.squeeze(0)  # (B, C, H, W)
            t = t.squeeze(0)  # (B)
            s = s.squeeze(0)  # (B)

            u = torch.zeros_like(t).to(self.device)
            pred_s = self._forward(
                self.decoder,
                noisy_xs_t,
                t,
                s,
                external_cond=z,
            )
            if self.dec_infer_steps > 1:
                pred_u = self._forward(
                    self.decoder,
                    noisy_xs_s,
                    s,
                    u,
                    external_cond=z,
                )

            if self.last_frame_loss_only:
                loss_s = F.mse_loss(
                    pred_s[-1:], noisy_xs_s[-1:].detach(), reduction="none"
                )
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )[-1:]
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u[-1:], xs[-1:].detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_u.ndim - 2))
                    )[-1:]
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()
            else:
                loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 1))
                )
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_s.ndim - 1))
                    )
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()

            self.log("training/rec_loss", loss)
            output_dict = {
                "loss": loss,
            }
            return output_dict
        return output_dict

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Test step of the model"""
        return self.validation_step(*args, **kwargs, namespace="test")  # type: ignore

    def on_test_epoch_end(self) -> None:
        """Operations when the test epoch ends"""
        self.on_validation_epoch_end(namespace="test")

    def on_validation_epoch_end(self, namespace: str = "validation") -> None:
        """Operations when the validation epoch ends"""
        if not self.validation_step_outputs:
            return
        xs_pred_ls = []
        xs_ls = []
        for pred, gt in self.validation_step_outputs:
            xs_pred_ls.append(pred)
            xs_ls.append(gt)
        xs_pred = torch.cat(xs_pred_ls, 1)
        xs = torch.cat(xs_ls, 1)

        if self.logger:
            log_video(
                xs_pred,
                xs.clone(),
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=0,
                logger=self.logger.experiment,
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred,
            xs,
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=self.validation_fvd_model,
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()

    def on_train_start(self) -> None:
        """Start tracing memory allocations"""
        tracemalloc.start()
        self.tracemalloc_snapshot = tracemalloc.take_snapshot()
