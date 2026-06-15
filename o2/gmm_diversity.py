"""
gmm_diversity.py
----------------
Differentiable GMM-based diversity loss for DCEM.

To use: import _fit_gmm and replace the log-det block in DCEM's
last-iteration diversity calculation with:

    gmm_loss, gmm_metrics = _fit_gmm(seq, K=2, n_iters=5)
    log_det_loss = gmm_loss
    diversity = {'action_var': action_var, **gmm_metrics}

And in training_utils.py rename the unpacked variable to gmm_loss.
"""

import torch


def _fit_gmm(seq, K=2, n_iters=5):
    """
    Fit a K-component GMM to decoded action samples via differentiable EM.

    Runs entirely in PyTorch so gradients flow back through the EM updates
    to the input samples.

    Args:
        seq:     [H, B, N, A] decoded action samples from the last CEM iteration.
        K:       number of GMM components.
        n_iters: number of EM iterations.

    Returns:
        loss:    scalar — negative weighted sum of log|Σ_k| (minimise = maximise diversity).
        metrics: dict of non-differentiable monitoring scalars.
    """
    H, B, N, A = seq.shape
    dev = seq.device
    reg = 1e-4 * torch.eye(A, device=dev)

    # Initialise mu by splitting samples evenly across components
    chunk = N // K
    mu = torch.stack(
        [seq[:, :, k * chunk:(k + 1) * chunk].mean(dim=2) for k in range(K)],
        dim=2,
    )  # [H, B, K, A]

    # Initialise Sigma as global empirical covariance
    global_mean = seq.mean(dim=2, keepdim=True)               # [H, B, 1, A]
    d0 = seq - global_mean                                     # [H, B, N, A]
    Sigma = (d0.unsqueeze(-1) * d0.unsqueeze(-2)).mean(dim=2) # [H, B, A, A]
    Sigma = Sigma.unsqueeze(2).expand(H, B, K, A, A).clone()  # [H, B, K, A, A]

    pi = seq.new_full((H, B, K), 1.0 / K)  # [H, B, K]

    TWO_PI = torch.tensor(2 * torch.pi, device=dev)

    for _ in range(n_iters):
        # ── E-step ────────────────────────────────────────────────────────────
        diff = seq.unsqueeze(3) - mu.unsqueeze(2)              # [H, B, N, K, A]
        Sigma_reg = Sigma + reg                                 # [H, B, K, A, A]

        L       = torch.linalg.cholesky(Sigma_reg)             # [H, B, K, A, A]
        log_det = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # [H, B, K]

        L_exp    = L.unsqueeze(2).expand(H, B, N, K, A, A)
        diff_col = diff.unsqueeze(-1)                          # [H, B, N, K, A, 1]
        v        = torch.linalg.solve_triangular(L_exp, diff_col, upper=False)
        mahal    = (v * v).sum(dim=-2).squeeze(-1)             # [H, B, N, K]

        log_lik = -0.5 * (A * TWO_PI.log() + log_det.unsqueeze(2) + mahal)
        log_r   = log_lik + (pi + 1e-8).log().unsqueeze(2)
        r       = torch.softmax(log_r, dim=-1)                # [H, B, N, K]

        # ── M-step ────────────────────────────────────────────────────────────
        N_k = r.sum(dim=2).clamp(min=1e-8)                   # [H, B, K]
        pi  = N_k / N

        mu  = (r.unsqueeze(-1) * seq.unsqueeze(3)).sum(dim=2) \
              / N_k.unsqueeze(-1)                             # [H, B, K, A]

        diff  = seq.unsqueeze(3) - mu.unsqueeze(2)            # [H, B, N, K, A]
        r_exp = r.unsqueeze(-1).unsqueeze(-1)                 # [H, B, N, K, 1, 1]
        Sigma = (r_exp * diff.unsqueeze(-1) * diff.unsqueeze(-2)).sum(dim=2) \
                / N_k.unsqueeze(-1).unsqueeze(-1)             # [H, B, K, A, A]

    # ── Diversity loss ─────────────────────────────────────────────────────────
    log_dets = torch.linalg.slogdet(Sigma + reg)[1]           # [H, B, K]
    loss     = -(pi * log_dets).sum(dim=-1).mean()            # scalar

    with torch.no_grad():
        metrics = {
            'gmm_diversity':  -loss.item(),
            'gmm_pi_balance': pi.min(dim=-1).values.mean().item(),
        }
        for k in range(K):
            metrics[f'gmm_mu_{k}_norm']     = mu[:, :, k].norm(dim=-1).mean().item()
            metrics[f'gmm_sigma_{k}_trace'] = Sigma[:, :, k].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
        if K == 2:
            metrics['gmm_inter_spread'] = \
                (mu[:, :, 0] - mu[:, :, 1]).norm(dim=-1).mean().item()

    return loss, metrics
