import time
from typing import Callable

import torch


def cubic_spline_nd_torch_batched(points: torch.Tensor) -> torch.Tensor:
    """Compute cubic splines

    Compute the second-derivatives (M) for a natural cubic spline in D dimensions,
    in a *batched* manner, without looping over batch or dimension axes.
    The only loop is over the knot index K for the Thomas algorithm.

    Parameters
    ----------
    points : torch.Tensor
        A (B, K, D) tensor of data points, where:
          B = batch size (number of spline problems),
          K = number of knots,
          D = dimension of each point.

    Returns
    -------
    M : torch.Tensor
        A (B, K, D) tensor of second derivatives for each batch and dimension.
        Natural boundary conditions (M[..., 0] = M[..., -1] = 0).
    """
    B, K, D = points.shape

    # If K <= 2, all second derivatives are zero.
    if K <= 2:
        return torch.zeros_like(points)

    # 1) Flatten from (B, K, D) => (B*D, K)
    #    Each row in the flattened array corresponds to one 1D spline problem.
    points_flat = points.permute(0, 2, 1)  # shape (B, D, K)
    points_flat = points_flat.reshape(-1, K)  # shape (B*D, K)

    # We'll solve for M_flat in shape (B*D, K).
    M_flat = torch.zeros_like(points_flat)  # same shape as points_flat

    # 2) Build alpha = 6*(y[i+1] - 2*y[i] + y[i-1]) for i in [1..K-2]
    #    We'll put alpha in a (B*D, K) array, with alpha[:, 0] and alpha[:, K-1] unused
    alpha = torch.zeros_like(points_flat)
    # Vectorized assignment for i=1..K-2
    alpha[:, 1 : K - 1] = 6.0 * (
        points_flat[:, 2:] - 2.0 * points_flat[:, 1:-1] + points_flat[:, :-2]
    )

    # 3) Prepare arrays l, mu, z of shape (B*D, K) for the Thomas algorithm
    l = torch.zeros_like(points_flat)  # noqa
    mu = torch.zeros_like(points_flat)
    z = torch.zeros_like(points_flat)

    # Boundary conditions: M[0] = 0 => l[0] = 1, z[0] = 0
    l[:, 0] = 1.0
    mu[:, 0] = 0.0
    z[:, 0] = 0.0

    # 4) Decomposition pass (loop over K dimension only)
    for i in range(1, K - 1):
        l[:, i] = 4.0 - mu[:, i - 1]
        mu[:, i] = 1.0 / l[:, i]
        z[:, i] = (alpha[:, i] - z[:, i - 1]) / l[:, i]

    # Boundary at the end
    l[:, K - 1] = 1.0
    z[:, K - 1] = 0.0

    # 5) Back-substitution pass
    for i in range(K - 2, 0, -1):
        M_flat[:, i] = z[:, i] - mu[:, i] * M_flat[:, i + 1]

    # M_flat[:, 0] and M_flat[:, K-1] remain zero => natural boundary
    #   (which is already the case by default initialization).

    # 6) Reshape back to (B, K, D)
    M = M_flat.view(B, D, K).permute(0, 2, 1)  # => (B, K, D)
    return M


def eval_cubic_spline_nd_torch_batched(
    points: torch.Tensor, M: torch.Tensor, ts: torch.Tensor
) -> torch.Tensor:
    """Evaluate batched natural cubic splines

    Evaluate batched natural cubic splines (in D dimensions) at multiple parameters,
    without looping over batch or dimension.

    Parameters
    ----------
    points : (B, K, D) torch.Tensor
        The knot points for each of the B splines.
    M : (B, K, D) torch.Tensor
        The second derivatives for each spline (same shape as points).
    ts : (B, N) torch.Tensor
        Each row has N query parameters at which to evaluate the corresponding spline.

    Returns
    -------
    vals : (B, N, D) torch.Tensor
        Spline evaluations. For each batch b, we evaluate the b-th spline
        at all the ts[b, :], yielding shape (N, D). Stacked into shape (B, N, D).
    """
    B, K, D = points.shape
    # Edge case: if K == 1, everything collapses to the single point
    if K == 1:
        # shape => (B, 1, D) repeated N times => (B, N, D)
        return points[:, 0:1, :].expand(B, ts.shape[1], D)

    # 1) Find the interval indices: i = floor(ts)
    #    Then clamp them so 0 <= i <= K-2
    i = torch.floor(ts).long()  # shape (B, N)
    i_clamped = torch.clamp(i, min=0, max=K - 2)

    # 2) Compute mu = fractional part
    mu = ts - i_clamped.float()  # shape (B, N)

    # 3) Gather y_i, y_{i+1}, M_i, M_{i+1} using advanced indexing
    #    We build an index of shape (B, N, D) to select from dimension=1 of points.
    gather_idx = i_clamped.unsqueeze(-1).expand(-1, -1, D)  # (B, N, D)

    # shape (B, N, D)
    y_i = torch.gather(points, dim=1, index=gather_idx)
    y_ip1 = torch.gather(points, dim=1, index=gather_idx + 1)
    M_i = torch.gather(M, dim=1, index=gather_idx)
    M_ip1 = torch.gather(M, dim=1, index=gather_idx + 1)

    # 4) Broadcast mu from (B, N) to (B, N, 1) so arithmetic aligns along D
    mu_3d = mu.unsqueeze(-1)  # (B, N, 1)
    one_minus_mu_3d = 1.0 - mu_3d

    # 5) Apply the natural cubic spline formula (uniform spacing h=1):
    #    S(t) = ((1 - mu)^3 * M_i + mu^3 * M_ip1) / 6
    #           + (y_i - M_i / 6) * (1 - mu)
    #           + (y_ip1 - M_ip1 / 6) * mu
    #    All shapes => (B, N, D). No Python loop over B or D.

    mu_cubed = mu_3d**3
    one_minus_mu_cubed = one_minus_mu_3d**3

    term_1 = (one_minus_mu_cubed * M_i + mu_cubed * M_ip1) / 6.0
    term_2 = (y_i - M_i / 6.0) * one_minus_mu_3d
    term_3 = (y_ip1 - M_ip1 / 6.0) * mu_3d

    vals = term_1 + term_2 + term_3  # shape (B, N, D)
    return vals


def cubic_spline_nd_function_torch(
    points: torch.Tensor,
) -> Callable[[float], torch.Tensor]:
    """Create a function to evaluate a natural cubic spline at any parameter t

    Given a set of points in D dimensions, precompute the second derivatives
    for a natural cubic spline, and return a function that can evaluate the spline
    at any parameter t in [0, N-1].

    Args:
        points: (B, K, D) tensor of K points in D dimensions, for B separate spline
        problems.

    Returns:
        A function that takes a parameter t in [0, N-1] and returns the spline value at
    """
    # 1) Precompute second derivatives in each dimension
    M = cubic_spline_nd_torch_batched(points)

    # 2) Return a closure that evaluates at any t in [0, N-1]
    def spline_func(t: float) -> torch.Tensor:
        return eval_cubic_spline_nd_torch_batched(points, M, t)

    return spline_func


# -------------------------------------------------------------------------
#                  EXAMPLE USAGE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we have B=2 separate spline problems, each with K=5 points in D=2.
    # So 'points' is (2, 5, 2):
    pts_2d = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, 1.0]],  # batch b=0
            [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 5.0], [4.0, 2.0]],  # batch b=1
        ],
        dtype=torch.float32,
    )
    pts_2d = pts_2d.repeat(50, 1, 1)

    ts = torch.linspace(0, 4, 100).view(1, -1).repeat(100, 1)

    s_time = time.time()
    # # 1) Compute batched second-derivatives: shape (B, K, D)
    # M_batched = cubic_spline_nd_torch_batched(pts_2d)

    # # 2) Evaluate all queries in a single call
    # vals_batched = eval_cubic_spline_nd_torch_batched(pts_2d, M_batched, ts)
    splin_func = cubic_spline_nd_function_torch(pts_2d)
    vals = splin_func(ts)

    # print("Batched second derivatives (B, K, D):\n", M_batched[:2], "\n")
    # print("Query parameters (B, N):\n", ts[:2], "\n")
    print("Spline evaluations (B, N, D):\n", vals[:2])
    print("Time taken:", time.time() - s_time)
