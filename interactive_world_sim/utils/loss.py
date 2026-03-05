import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def find_match(
    latent_1: torch.Tensor, latent_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find match between two image latents using Hungarian algorithm.

    Args:
        latent_1: Latent representation of the first image (B, C, H, W)
        latent_2: Latent representation of the second image (B, C, H, W)

    Returns:
        Match between the two latents (B, H*W, 2), (B, H*W, 2)
    """
    # distance in latent space
    B, C, H, W = latent_1.shape
    latent_1 = latent_1.view(B, C, -1)  # (B, C, H*W)
    latent_1 = latent_1.permute(0, 2, 1)  # (B, H*W, C)
    latent_2 = latent_2.view(B, C, -1)  # (B, C, H*W)
    latent_2 = latent_2.permute(0, 2, 1)  # (B, H*W, C)
    dist = torch.cdist(latent_1, latent_2, p=2)  # (B, H*W, H*W)

    # distance in pixel space
    coord = torch.meshgrid(torch.arange(H), torch.arange(W))  # (H, W), (H, W)
    coord = torch.stack(coord, dim=-1)  # (H, W, 2) in (row, col)
    coord = coord.to(latent_1.device)  # (H, W, 2)
    coord = coord.view(-1, 2)  # (H*W, 2)
    coord[:, 0] = coord[:, 0] / (H - 1)  # normalize to [0, 1]
    coord[:, 1] = coord[:, 1] / (W - 1)  # normalize to [0, 1]
    coord = coord.unsqueeze(0).to(torch.float)  # (1, H*W, 2)
    coord_dist = torch.cdist(coord, coord, p=2)  # (1, H*W, H*W)

    total_dist = dist + 0.5 * coord_dist  # (B, H*W, H*W)

    # cpu version
    latent1_matches = []
    latent2_matches = []
    for b in range(B):
        dist_np = total_dist[b].detach().cpu().numpy()
        latent1_ind, latent2_ind = linear_sum_assignment(dist_np, maximize=False)
        latent1_matches.append(latent1_ind)
        latent2_matches.append(latent2_ind)
    latent1_idx_np = np.stack(latent1_matches, axis=0)  # (B, H*W)
    latent2_idx_np = np.stack(latent2_matches, axis=0)  # (B, H*W)
    latent1_idx_tensor = torch.from_numpy(latent1_idx_np).to(
        latent_1.device
    )  # (B, H*W)
    latent2_idx_tensor = torch.from_numpy(latent2_idx_np).to(
        latent_2.device
    )  # (B, H*W)

    # # cuda version
    # assignment = batch_linear_assignment(total_dist)
    # latent1_idx_tensor = torch.arange(H * W, device=latent_1.device).unsqueeze(0)
    # latent1_idx_tensor = latent1_idx_tensor.expand(B, -1)
    # latent2_idx_tensor = assignment

    latent1_idx_row = latent1_idx_tensor // W
    latent1_idx_col = latent1_idx_tensor % W
    latent2_idx_row = latent2_idx_tensor // W
    latent2_idx_col = latent2_idx_tensor % W
    match1 = torch.stack([latent1_idx_row, latent1_idx_col], dim=-1)  # (B, H*W, 2)
    match2 = torch.stack([latent2_idx_row, latent2_idx_col], dim=-1)  # (B, H*W, 2)
    return match1, match2


def emd_loss(latent_1: torch.Tensor, latent_2: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance (EMD) loss between two image latents.

    Args:
        latent_1: Latent representation of the first image (B, C, H, W)
        latent_2: Latent representation of the second image (B, C, H, W)

    Returns:
        EMD loss between the two latents (B,)
    """
    H, W = latent_1.shape[2:]
    latent_1_idx, latent_2_idx = find_match(latent_1, latent_2)
    latent_1_idx = latent_1_idx.to(torch.float32)
    latent_1_idx[..., 0] = latent_1_idx[..., 0] / (H - 1)
    latent_1_idx[..., 1] = latent_1_idx[..., 1] / (W - 1)
    latent_2_idx = latent_2_idx.to(torch.float32)
    latent_2_idx[..., 0] = latent_2_idx[..., 0] / (H - 1)
    latent_2_idx[..., 1] = latent_2_idx[..., 1] / (W - 1)
    loss = torch.nn.functional.mse_loss(latent_1_idx, latent_2_idx, reduction="none")
    loss = loss.mean(dim=-1).mean(dim=-1)
    return loss
