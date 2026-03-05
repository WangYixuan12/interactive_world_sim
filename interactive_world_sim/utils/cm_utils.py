import math
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Beta


@torch.no_grad()
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


@torch.no_grad()
def reduce_dims(x, target_dims):
    """Reduces dimensions from the end of a tensor until it has target_dims dimensions."""
    dims_to_reduce = x.ndim - target_dims
    if dims_to_reduce < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is greater"
        )
    for _ in range(dims_to_reduce):
        x = x.squeeze(-1)

    return x


class Karras_Scheduler:
    def __init__(
        self,
        time_min,
        time_max,
        rho,
        bins,
        solver,
        time_sampler,
        scaling,
        data_std,
        # log normal sampling
        P_std,
        P_mean,
        weighting="none",
        # euler marayuma corrector step
        alpha=-1,  # ratio btwn corr step and ode step
        friction=-1,  # friction for corr step
        # time chunking
        beta=0.0,
        name="singular",
        **kwargs,  # for compatibility with old code, should clean up later
    ):

        self.time_min = time_min
        self.time_max = time_max
        self.rho = rho
        self.bins = bins
        self.data_std = data_std

        self.solver = solver
        self.time_sampler = time_sampler
        self.scaling = scaling

        self.weighting = weighting

        # log normal sampling
        self.P_std = P_std
        self.P_mean = P_mean

        # langevin corrector step
        self.alpha = alpha
        self.friction = friction

        # time chunking
        self.beta = beta
        self.bins_min = 0
        self.bins_max = bins

        self.name = name
        print("Using scheduler {}".format(self.name))

        if "corr" in self.solver:
            if self.alpha < 0:
                raise ValueError("alpha must be specified for corr solver")
            if self.friction < 0:
                raise ValueError("friction must be specified for corr solver")

    # ==================== MAIN ====================
    def step(self, model, samples, t, next_t, clamp=False):
        if self.solver == "euler" or self.solver == "first_order":
            return self.euler_solver(model, samples, t, next_t, clamp=clamp)
        elif self.solver == "heun" or self.solver == "second_order":
            return self.heun_solver(model, samples, t, next_t, clamp=clamp)
        elif self.solver == "third":
            return self.third_order_solver(model, samples, t, next_t, clamp=clamp)
        elif self.solver == "fourth":
            return self.fourth_order_solver(model, samples, t, next_t, clamp=clamp)
        elif self.solver == "second_order_corr":
            return self.second_order_corr_solver(model, samples, t, next_t, clamp=clamp)
        else:
            raise ValueError(f"Unknown solver {self.solver}")

    def calc_out(
        self, model, trajectory: torch.Tensor, times: torch.Tensor, clamp=False
    ):
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings_for_boundary_condition(times)
            ]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim) for c in self.get_scalings(times)
            ]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        rescaled_times = (
            1000 * 0.25 * torch.log(times + 1e-44)
        )  # *1000 to make it more stable
        model_output = model(trajectory * c_in, rescaled_times)

        out = model_output * c_out + trajectory * c_skip
        if clamp:
            out = out.clamp(-1.0, 1.0)  # this should only happen at inference time

        return out

    def add_noise(self, trajectory: torch.Tensor, times: torch.Tensor):
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        return trajectory + self.trajectory_time_product(noise, times)

    def sample_inital_position(self, trajectory, generator):

        traj = torch.randn(
            size=trajectory.shape,
            dtype=trajectory.dtype,
            device=trajectory.device,
            generator=generator,
        )

        return traj  ## * self.time_max ## Reducing Initial Variance by not multiplying by time_max

    # ==================== TIME SAMPLERS ====================

    # def sample_times(self, trajectory: torch.Tensor, time_sampler = None):
    #     time_sampler = time_sampler if time_sampler is not None else self.time_sampler
    #     batch = trajectory.shape[0]
    #     device = trajectory.device

    #     if time_sampler == "uniform":
    #         return self.uniform_sampler(batch, device)
    #     elif time_sampler == "log_normal":
    #         return self.log_normal_sampler(batch, device)
    #     elif time_sampler == "uniform_time_chunked":
    #         return self.uniform_time_chunked_sampler(batch, device)
    #     elif time_sampler == "ctm_dsm":
    #         return torch.cat((self.ctm_dsm_sampler(int(math.ceil(batch/2)), device)[0], self.log_normal_sampler(int(math.floor(batch/2)), device)[0]), dim = 0), None
    #     else:
    #         raise ValueError(f"Unknown sampler {time_sampler}")

    def sample_times(
        self, xs: torch.Tensor, time_sampler: Optional[str] = None
    ) -> torch.Tensor:
        """Generate noise levels for training."""
        time_sampler = time_sampler if time_sampler is not None else self.time_sampler
        num_frames, batch_size, *_ = xs.shape
        match time_sampler:
            case "random_last":  # random noise levels for the last frame
                last_noise_levels = torch.randint(
                    0,
                    self.bins,
                    (1, batch_size),
                    device=xs.device,
                )
                prev_noise_levels = torch.randint(
                    0,
                    int(self.bins * 0.1),
                    (num_frames - 1, batch_size),
                    device=xs.device,
                )
                noise_levels = torch.cat([prev_noise_levels, last_noise_levels], 0)
                times = self.timesteps_to_times(noise_levels)
            case "log_normal":
                last_times = (
                    torch.randn((1, batch_size), device=xs.device) * self.P_std
                    + self.P_mean
                ).exp()
                prev_noise_levels = torch.randint(
                    0,
                    int(self.bins * 0.1),
                    (num_frames - 1, batch_size),
                    device=xs.device,
                )
                prev_times = self.timesteps_to_times(prev_noise_levels)
                times = torch.cat([prev_times, last_times], 0)

        return times

    def uniform_sampler(self, batch, device):
        timesteps = torch.randint(
            0,
            self.bins - 1,
            (batch,),
            device=device,
        ).long()

        return self.timesteps_to_times(timesteps), self.timesteps_to_times(
            timesteps + 1
        )

    def uniform_time_chunked_sampler(self, batch, device):

        # these should all cross the boundary
        if random.random() < self.beta:
            # timesteps should just all be bins_max
            timesteps = (
                torch.ones(
                    (int(batch),),
                    device=device,
                ).long()
                * self.bins_max
            )

            times, next_times = self.timesteps_to_times(
                timesteps
            ), self.timesteps_to_times(timesteps - 1)
            return times, next_times

        # these should all be inside the current chunk
        timesteps = torch.randint(
            self.bins_min,
            self.bins_max - 1,
            (batch,),
            device=device,
        ).long()

        return self.timesteps_to_times(timesteps), self.timesteps_to_times(
            timesteps + 1
        )

    def get_boundary(self):
        """Returns the lower time boundary of the current chunk, which corresponds to the max bin"""
        return self.timesteps_to_times(torch.tensor(self.bins_max))

    def log_normal_sampler(self, batch, device):
        """Sample times such that mean and variance of the ln of the times is -1.2, 1.2.
        Goal is to target training towards the beginning of the diffusion process
        """
        sigma = (torch.randn((batch,), device=device) * self.P_std + self.P_mean).exp()

        return sigma, sigma

    def ctm_dsm_sampler(self, batch, device):
        sigma_max = self.time_max
        sigma_min = self.time_min
        ro = self.rho

        # Generate random samples uniformly from the interval [0, 0.7]
        xi_samples = torch.rand((batch,), device=device) * 0.7
        # Apply the transformation to these samples
        transformed_samples = (
            sigma_max ** (1 / ro)
            + xi_samples * (sigma_min ** (1 / ro) - sigma_max ** (1 / ro))
        ) ** ro
        transformed_samples = transformed_samples.clamp(sigma_min, sigma_max)
        return transformed_samples, transformed_samples

    def timesteps_to_times(self, timesteps: torch.LongTensor):
        min_inv_rho = self.time_min ** (1 / self.rho)
        max_inv_rho = self.time_max ** (1 / self.rho)
        t = min_inv_rho + timesteps / (self.bins - 1) * (max_inv_rho - min_inv_rho)
        t = t**self.rho

        return t.clamp(self.time_min, self.time_max)

    def times_to_timesteps(self, times: torch.Tensor):
        min_inv_rho = self.time_min ** (1 / self.rho)
        max_inv_rho = self.time_max ** (1 / self.rho)
        times_inv_rho = times ** (1 / self.rho)
        timesteps = (
            (times_inv_rho - min_inv_rho)
            * (self.bins - 1)
            / (max_inv_rho - min_inv_rho)
        )

        timesteps = torch.round(timesteps)
        return timesteps.long()

    # ==================== WEIGHTINGS ====================
    def get_weights(self, times, next_times, weighting=None):
        """Returns weights to scale loss by.
        Currently supports ICT and Karras weighting
        """
        weighting = weighting if weighting is not None else self.weighting

        if weighting == "none":
            return None
        elif weighting == "ict":
            return self.get_ict_weightings(times, next_times)
        elif weighting == "karras":
            return self.get_karras_weightings(times)
        else:
            raise ValueError(f"Unknown weighting {weighting}")

    def get_ict_weightings(self, times, next_times):
        return 1 / (times - next_times)

    def get_karras_weightings(self, times, **kwargs):
        return (times**2 + self.data_std**2) / ((times * self.data_std) ** 2)

    # ==================== PARAMETIRIZATIONS ====================
    def get_scalings(self, time):
        c_skip = self.data_std**2 / (time**2 + self.data_std**2)
        c_out = time * self.data_std / ((time**2 + self.data_std**2) ** 0.5)
        c_in = 1 / (time**2 + self.data_std**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, time):
        c_skip = self.data_std**2 / ((time - self.time_min) ** 2 + self.data_std**2)
        c_out = (
            (time - self.time_min) * self.data_std / (time**2 + self.data_std**2) ** 0.5
        )
        c_in = 1 / (time**2 + self.data_std**2) ** 0.5
        return c_skip, c_out, c_in

    # ==================== SOLVERS ====================
    @torch.no_grad()
    def euler_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        y = samples
        step = append_dims((next_t - t), dims)

        denoisedy = self.calc_out(model, y, t, clamp=clamp)
        dy = (y - denoisedy) / append_dims(t, dims)

        y_next = samples + step * dy

        return y_next

    @torch.no_grad()
    def heun_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        y = samples
        step = append_dims((next_t - t), dims)

        denoisedy = self.calc_out(model, y, t, clamp=clamp)
        dy = (y - denoisedy) / append_dims(t, dims)

        y_next = samples + step * dy

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoisedy_next) / append_dims(next_t, dims)

        y_next = samples + step * (dy + dy_next) / 2

        return y_next

    @torch.no_grad()
    def third_order_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        y = samples
        step = next_t - t

        denoisedy = self.calc_out(model, y, t, clamp=clamp)
        dy = (y - denoisedy) / append_dims(t, dims)

        y_next = samples + append_dims(step, dims) * dy

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoisedy_next) / append_dims(next_t, dims)

        y_mid = samples + append_dims(step, dims) / 2 * (dy + dy_next) / 2

        denoisedy_mid = self.calc_out(model, y_mid, t + step / 2, clamp=clamp)
        dy_mid = (y_mid - denoisedy_mid) / append_dims(t + step / 2, dims)

        dy_final = (dy + 4 * dy_mid + dy_next) / 6
        y_third = samples + append_dims(step, dims) * dy_final

        return y_third

    @torch.no_grad()
    def fourth_order_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        y = samples
        step = next_t - t

        pred = self.calc_out(model, y, t, clamp=clamp)
        k_1 = (y - pred) / append_dims(t, dims)

        y_2 = samples + append_dims(step, dims) / 2 * k_1
        pred = self.calc_out(model, y_2, t + step / 2, clamp=clamp)
        k_2 = (y_2 - pred) / append_dims(t + step / 2, dims)

        y_3 = samples + append_dims(step, dims) / 2 * k_2
        pred = self.calc_out(model, y_3, t + step / 2, clamp=clamp)
        k_3 = (y_3 - pred) / append_dims(t + step / 2, dims)

        y_4 = samples + append_dims(step, dims) * k_3
        pred = self.calc_out(model, y_4, t + step, clamp=clamp)
        k_4 = (y_4 - pred) / append_dims(t + step, dims)

        y_next = samples + append_dims(step, dims) / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

        return y_next

    @torch.no_grad()
    def second_order_corr_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        x_0 = self.heun_solver(
            model, samples, t, t + (next_t - t) * (1 - self.alpha), clamp=clamp
        )
        t_0 = t + (next_t - t) * (1 - self.alpha)
        step = append_dims((next_t - t_0), dims)
        # sample v_0 from guassian noise
        v_0 = torch.randn_like(samples)
        # sample brownian noise
        w_t = torch.randn_like(samples)

        # integrate velocity
        drift = self.calc_out(model, x_0, t_0, clamp=clamp) - v_0 * self.friction
        diffusion = w_t * (2 * self.friction) ** 0.5
        v_t = (
            v_0 + drift * step + diffusion * step.abs().sqrt()
        )  # abs because step is negative, but we don't have to change the sign of + diffusion b/c diffusion is a guassian

        # integrate position
        x_t = x_0 + (v_0 + v_t) / 2 * step

        return x_t

    # ==================== HELPERS ====================
    @staticmethod
    def trajectory_time_product(traj: torch.Tensor, times: torch.Tensor):
        return traj * times[..., None, None, None]


class PFGMPP_Scheduler(Karras_Scheduler):
    def __init__(
        self,
        time_min,
        time_max,
        rho,
        bins,
        solver,
        time_sampler,
        scaling,
        data_std,
        # log normal sampling
        P_std,
        P_mean,
        weighting="none",
        # euler marayuma corrector step
        alpha=-1,  # ratio btwn corr step and ode step
        friction=-1,  # friction for corr step
        # pfgm++
        D=-1,
        N=-1,
        # time chunking
        beta=0.0,
        name="singular",
        **kwargs,  # for compatibility with old code, should clean up later):
    ):
        super().__init__(
            time_min=time_min,
            time_max=time_max,
            rho=rho,
            bins=bins,
            solver=solver,
            time_sampler=time_sampler,
            scaling=scaling,
            data_std=data_std,
            P_std=P_std,
            P_mean=P_mean,
            weighting=weighting,
            alpha=alpha,
            friction=friction,
            beta=beta,
            name=name,
        )
        self.D = D
        if self.D == -1:
            raise ValueError("D must be specified for pfgmpp")

        self.N = N
        if self.N == -1:
            raise ValueError("N must be specified for pfgmpp")

        print("Using pfgmpp with D = {}, N = {}".format(self.D, self.N))

    def sample_inital_position(self, trajectory, generator):
        def rand_beta_prime(size, device, N, D):
            # sample from beta_prime (N/2, D/2)
            beta_gen = Beta(torch.FloatTensor([N / 2.0]), torch.FloatTensor([D / 2.0]))

            sample_norm = beta_gen.sample().to(device).double()
            # inverse beta distribution
            inverse_beta = sample_norm / (1 - sample_norm)

            sample_norm = torch.sqrt(inverse_beta) * self.time_max * np.sqrt(D)
            gaussian = torch.randn(size[0], N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            traj = unit_gaussian * sample_norm

            return traj.view(size)

        if self.N != trajectory.shape[-1] * trajectory.shape[-2]:
            raise ValueError(
                "N must be equal to T * D for pfgmpp but N is {} and T * D is {}".format(
                    self.N, trajectory.shape[-1] * trajectory.shape[-2]
                )
            )

        traj = rand_beta_prime(
            trajectory.shape,
            trajectory.device,
            N=self.N,
            D=self.D,
        )

        return traj

    def add_noise(self, trajectory: torch.Tensor, times: torch.Tensor):
        if self.N != trajectory.shape[-1] * trajectory.shape[-2]:
            raise ValueError(
                "N must be equal to T * D for pfgmpp but N is {} and T * D is {}".format(
                    self.N, trajectory.shape[-1] * trajectory.shape[-2]
                )
            )

        r = times.double() * np.sqrt(self.D).astype(np.float64)

        # Sampling from inverse-beta distribution
        samples_norm = np.random.beta(
            a=self.N / 2.0, b=self.D / 2.0, size=trajectory.shape[0]
        ).astype(np.double)

        samples_norm = np.clip(samples_norm, 1e-3, 1 - 1e-3)

        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        inverse_beta = torch.from_numpy(inverse_beta).to(trajectory.device).double()
        # Sampling from p_r(R) by change-of-variable
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = samples_norm.view(len(samples_norm), -1)
        # Uniformly sample the angle direction
        gaussian = torch.randn(trajectory.shape[0], self.N).to(samples_norm.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * samples_norm
        perturbation_x = perturbation_x.float()

        n = perturbation_x.view_as(trajectory)  # B X (T X D) -> B X T X D
        return trajectory + n


class CTM_Scheduler(Karras_Scheduler):
    def __init__(
        self,
        time_min,
        time_max,
        rho,
        bins,
        solver,
        time_sampler,
        scaling,
        data_std,
        # log normal sampling
        P_std,
        P_mean,
        weighting="none",
        # euler marayuma corrector step
        alpha=-1,  # ratio btwn corr step and ode step
        friction=-1,  # friction for corr step
        # time chunking
        beta=0.0,
        # ctm
        ode_steps_max=-1,
        name="singular",
        **kwargs,  # for compatibility with old code, should clean up later):
    ):

        if ode_steps_max == -1:
            raise ValueError("ode_steps_max must be specified for CTM scheduler")

        self.ode_steps_max = ode_steps_max

        super().__init__(
            time_min=time_min,
            time_max=time_max,
            rho=rho,
            bins=bins,
            solver=solver,
            time_sampler=time_sampler,
            scaling=scaling,
            data_std=data_std,
            P_std=P_std,
            P_mean=P_mean,
            weighting=weighting,
            alpha=alpha,
            friction=friction,
            beta=beta,
            name=name,
        )
        print("Using CTM scheduler")

    def CTM_calc_out(
        self,
        model: Any,
        trajectory: torch.Tensor,
        times: torch.Tensor,
        stops: torch.Tensor,
        clamp: bool = False,
    ) -> torch.Tensor:
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings_for_boundary_condition(times)
            ]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim) for c in self.get_scalings(times)
            ]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        # if times.ndim > 1:
        #     times = reduce_dims(times, 1)

        # if stops.ndim > 1:
        #     stops = reduce_dims(stops, 1)

        rescaled_times = 1000 * 0.25 * torch.log(times + 1e-44)
        rescaled_stops = 1000 * 0.25 * torch.log(stops + 1e-44)

        model_output = model(trajectory * c_in, rescaled_times, rescaled_stops)

        out = model_output * c_out + trajectory * c_skip  # g_theta

        ratio = (stops / times).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = trajectory * ratio + out * (1 - ratio)  # G_theta

        if clamp:
            out = out.clamp(-1.0, 1.0)  # this should only happen at inference time

        return out

    # def sample_times(self, trajectory: torch.Tensor, time_sampler=None):
    #     time_sampler = time_sampler if time_sampler is not None else self.time_sampler
    #     batch = trajectory.shape[0]
    #     device = trajectory.device

    #     # this sampler returns t, s, u as bins
    #     if time_sampler == "ctm":
    #         # t is uniform over bins
    #         t = torch.randint(
    #             0,
    #             self.bins,
    #             (batch,),
    #             device=device,
    #         ).long()
    #         # s is uniform over bins greater than t
    #         s = torch.cat(
    #             [torch.randint(int(t_i.item()), self.bins + 1, (1,)) for t_i in t]
    #         ).to(device)

    #         # u is uniform over bins between t and s
    #         u = torch.cat(
    #             [
    #                 (torch.randint(int(t_i.item()), int((s_i + 1).item()), (1,)))
    #                 for t_i, s_i in zip(
    #                     t, s
    #                 )  # might want to swap bound vs clamp (s_i vs self.ode_max) depending on their relative distributions
    #             ]
    #         ).to(device)

    #         maxes = t + self.ode_steps_max
    #         mask = (u > maxes).float()
    #         u = u * (1 - mask) + maxes * mask

    #         return t, s, u

    #     if time_sampler == "ctm_to_cm_ln":
    #         # t is log normal with mean -1.2, std 1.2
    #         t, _ = self.log_normal_sampler(batch, device)
    #         t = self.times_to_timesteps(t)
    #         # s is min
    #         s = torch.tensor([self.time_min], device=device).expand(batch).long()
    #         # u is 1 less than t
    #         u = t + 1

    #         return t, s, u

    #     if time_sampler == "ctm_to_cm":
    #         # t is uniform over bins
    #         t = torch.randint(
    #             0,
    #             self.bins,
    #             (batch,),
    #             device=device,
    #         ).long()
    #         # s is min
    #         s = torch.tensor([self.time_min], device=device).expand(batch).long()
    #         # u is 1 less than t
    #         u = t + 1

    #         return t, s, u

    #     else:
    #         return super().sample_times(trajectory, time_sampler=time_sampler)

    def sample_times(self, xs: torch.Tensor, time_sampler=None):
        time_sampler = time_sampler if time_sampler is not None else self.time_sampler
        num_frames, batch_size, *_ = xs.shape
        device = xs.device

        # this sampler returns t, s, u as bins
        prev_t = torch.randint(
            0,
            int(self.bins * 0.1),
            (num_frames - 1, batch_size),
            device=xs.device,
        )
        if time_sampler == "ctm":
            # t is uniform over bins
            t = torch.randint(
                0,
                self.bins,
                (batch_size,),
                device=device,
            ).long()
            # s is uniform over bins greater than t
            s = torch.cat(
                [torch.randint(int(t_i.item()), self.bins + 1, (1,)) for t_i in t]
            ).to(device)

            # u is uniform over bins between t and s
            u = torch.cat(
                [
                    (torch.randint(int(t_i.item()), int((s_i + 1).item()), (1,)))
                    for t_i, s_i in zip(t, s, strict=False)
                ]
            ).to(device)

            s = torch.cat([prev_t, s[None]], 0)
            t = torch.cat([prev_t, t[None]], 0)
            u = torch.cat([prev_t, u[None]], 0)

            maxes = t + self.ode_steps_max
            mask = (u > maxes).float()
            u = u * (1 - mask) + maxes * mask
            return t, s, u
        elif time_sampler == "ctm_dsm":
            prev_t = self.timesteps_to_times(prev_t)
            t = torch.cat(
                (
                    self.ctm_dsm_sampler(int(math.ceil(batch_size / 2)), device)[0],
                    self.log_normal_sampler(int(math.floor(batch_size / 2)), device)[0],
                ),
                dim=0,
            )
            t = torch.cat([prev_t, t[None]], 0)
            return t, None
        else:
            return super().sample_times(xs, time_sampler=time_sampler)

    # CTM needs to handle step size 0
    @torch.no_grad()
    def heun_solver(self, model, samples, t, next_t, clamp=False):
        dims = samples.ndim
        y = samples
        step = append_dims((next_t - t), dims)
        mask = (step == 0).float()

        denoisedy = self.calc_out(model, y, t, clamp=clamp)
        dy = (y - denoisedy) / (append_dims(t, dims) + mask)

        y_next = samples + step * dy

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoisedy_next) / (append_dims(next_t, dims) + mask)

        y_next = samples + step * (dy + dy_next) / 2

        y_next = y_next * (1 - mask) + samples * mask

        return y_next


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a[t]
    dtype = out.dtype
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out + torch.zeros(x_shape, device=t.device, dtype=dtype)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """linear schedule, proposed in original ddpm paper"""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """cosine schedule

    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    """sigmoid schedule

    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        x_shape: torch.Size,
        timesteps: int,
        sampling_timesteps: int,
        beta_schedule: str,
        schedule_fn_kwargs: dict,
        objective: str,
        snr_clip: float,
        cum_snr_decay: float,
        ddim_sampling_eta: float,
        clip_noise: float,
        stabilization_level: int,
        dtype: torch.dtype = torch.float32,
        use_fused_snr: bool = True,
        loss_weighting: str = "fused_snr",
    ) -> None:
        super().__init__()
        self.x_shape = x_shape
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.dtype = dtype
        self.schedule_fn_kwargs = schedule_fn_kwargs
        self.objective = objective
        self.loss_weighting = loss_weighting
        self.snr_clip = snr_clip
        self.cum_snr_decay = cum_snr_decay
        self.ddim_sampling_eta = ddim_sampling_eta
        self.clip_noise = clip_noise
        self.stabilization_level = stabilization_level

        self._build_buffer()

    def _build_buffer(self) -> None:
        if self.beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {self.beta_schedule}")

        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)
        betas = betas.to(self.dtype)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.dtype)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(self.dtype)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        # register_buffer(
        #     "p2_loss_weight",
        #     (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
        #     ** -self.p2_loss_weight_gamma,
        # )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the Q posterior."""
        if noise is None:
            noise = torch.randn_like(x_start, dtype=self.dtype)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def add_noise(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, t=times, noise=noise)
        return noised_x

    def add_noise_return_tgt(
        self, x: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, t=times, noise=noise)
        if self.objective == "pred_v":
            target = self.predict_v(x, times, noise)
        return noised_x, target

    def add_noise_to_t_s(
        self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x_t = self.q_sample(x_start=x, t=t, noise=noise)
        noised_x_s = self.q_sample(x_start=x, t=s, noise=noise)
        return noised_x_t, noised_x_s

    def step(
        self,
        model: Any,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        clamp: bool = False,
    ) -> torch.Tensor:
        """DDIM sampling step."""
        assert (curr_noise_level[-1] > next_noise_level[-1]).all()
        assert (curr_noise_level >= next_noise_level).all()
        # convert noise level -1 to self.stabilization_level - 1
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(
                curr_noise_level, self.stabilization_level - 1, dtype=torch.long
            ),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x),
        )
        x = torch.where(
            self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x
        )

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        model_pred = model(
            x=x,
            t=clipped_curr_noise_level,
        )
        x_start = model_pred.pred_x_start
        pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )

        return x_pred

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise from the start."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict the v."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Predict the start from the v."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_weights(self, noise_levels: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels."""
        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        if self.loss_weighting == "max_snr":
            # min SNR reweighting
            clipped_snr_max = torch.clamp(snr, max=self.snr_clip)
            match self.objective:
                case "pred_noise":
                    return clipped_snr_max / snr
                case "pred_x0":
                    return clipped_snr_max
                case "pred_v":
                    return clipped_snr_max / (snr + 1)
        elif self.loss_weighting == "uniform":
            return torch.ones_like(noise_levels)
        elif self.loss_weighting == "min_snr":
            clipped_snr_min = torch.clamp(snr, min=self.snr_clip)
            match self.objective:
                case "pred_noise":
                    return clipped_snr_min / snr
                case "pred_x0":
                    return clipped_snr_min
                case "pred_v":
                    return clipped_snr_min / (snr + 1)
        elif self.loss_weighting == "fused_snr":
            cum_snr = torch.zeros_like(normalized_snr)
            for t in range(noise_levels.shape[0]):
                if t == 0:
                    cum_snr[t] = normalized_clipped_snr[t]
                else:
                    cum_snr[t] = (
                        self.cum_snr_decay * cum_snr[t - 1]
                        + (1 - self.cum_snr_decay) * normalized_clipped_snr[t]
                    )

            cum_snr = F.pad(cum_snr[:-1], (0, 0, 1, 0), value=0.0)
            clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (
                1 - normalized_clipped_snr
            )
            fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)

            match self.objective:
                case "pred_noise":
                    return clipped_fused_snr / fused_snr
                case "pred_x0":
                    return clipped_fused_snr * self.snr_clip
                case "pred_v":
                    return (
                        clipped_fused_snr
                        * self.snr_clip
                        / (fused_snr * self.snr_clip + 1)
                    )
                case _:
                    raise ValueError(f"unknown objective {self.objective}")

    def calc_out(
        self, model: Any, x: torch.Tensor, t: torch.Tensor, clamp: bool = False
    ) -> torch.Tensor:
        model_output = model(x, t)

        if self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
        else:
            raise ValueError(f"Unknown objective {self.objective}")

        return x_start

    def CTM_calc_out(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        clamp: bool = False,
    ) -> torch.Tensor:
        assert (t >= s).all()
        assert (t[-1] > s[-1]).all()
        model_output = model(x, t, s)

        if self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
        else:
            raise ValueError(f"Unknown objective {self.objective}")

        ratio = (s / t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = x * ratio + x_start * (1 - ratio)  # G_theta

        return out

    def model_prediction(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = model(x, t)

        if self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f"Unknown objective {self.objective}")

        return v, x_start, pred_noise

    def ddim_step(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ):
        assert (t >= 0).all()
        assert (s >= 0).all()
        assert (t >= s).all()
        assert (t[-1] > s[-1]).all()
        orig_x = x.clone()

        # infer x_start at t
        _, x_start, pred_noise = self.model_prediction(model, x, t)

        # step to s
        alpha = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[s]

        sigma = (
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = t == s
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )
        return x_pred

    def add_shape_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Add shape channels to the tensor."""
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")


class FlowMatchingScheduler(nn.Module):
    def __init__(
        self,
        sampling_strategy: str = "random",
    ) -> None:
        super().__init__()
        self.sampling_strategy = sampling_strategy

    def sample_times(self, x: torch.Tensor) -> torch.Tensor:
        """Sample times based on the sampling strategy.

        Args:
            x: Input tensor of shape (T, B, C, H, W)
        """
        T, B, C, H, W = x.shape
        if self.sampling_strategy == "random":
            last_t = torch.rand(1, B, device=x.device)  # shape (1, B)
            prev_t = torch.rand(T - 1, B, device=x.device) * (
                1 - 0.05
            )  # shape (T-1, B)
            t = torch.cat([prev_t, last_t], dim=0)  # shape (T, B)
            return t
        else:
            raise ValueError(f"Unknown sampling strategy {self.sampling_strategy}")

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to the input tensor based on times t.

        Args:
            x: Input tensor of shape (T, B, C, H, W)
            t: Times tensor of shape (T, B)
        """
        noise = torch.randn_like(x)
        dx = x - noise
        t = t.reshape(t.shape + (1, 1, 1))  # shape (T, B, 1, 1, 1)
        return t * x + (1 - t) * noise, dx

    @torch.no_grad()
    def step(
        self, model, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single step of the flow matching scheduler.

        Args:
            model: The model to use for prediction.
            x_t: Input tensor at time t of shape (T, B, C, H, W)
            t: Current times tensor of shape (T, B)
            s: Next times tensor of shape (T, B)
        """
        model_output = model(x_t, t)
        s_reshape = s.reshape(s.shape + (1, 1, 1))  # shape (T, B, 1, 1, 1)
        t_reshape = t.reshape(t.shape + (1, 1, 1))  # shape (T, B, 1, 1, 1)
        model_output_2 = model(
            x_t + model_output * (s_reshape - t_reshape) / 2.0, (t + s) / 2.0
        )

        return x_t + (s_reshape - t_reshape) * model_output_2


def Pseudo_Huber_Loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes psuedo-huber loss of pred and target of shape (batch, time, dim)"""
    pred = rearrange(pred, "t b c h w -> (t b) c h w")
    target = rearrange(target, "t b c h w -> (t b) c h w")
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    loss = torch.sqrt(mse + delta**2) - delta  # (t * b)
    if weights is not None:
        weights = rearrange(weights, "t b -> (t b)")
        loss = loss * weights
    return loss.mean()


def Huber_Loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes psuedo-huber loss of pred and target of shape (batch, time, dim)

    Delta is the boundary between l_1 and l_2 loss. At delta = 0, this is just MSE loss.
    Setting delta = -1 calculates iCT's recommended delta given data size.

    Also supports weighting of loss
    """
    mse = F.mse_loss(pred, target, reduction="none")

    loss = torch.sqrt(mse + delta**2) - delta

    if weights is not None:
        loss = loss * weights[..., None, None, None]

    return loss.mean()
