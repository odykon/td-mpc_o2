"""
gmm_diversity.py
----------------
Detached GMM-based diversity loss for DCEM.

EM is run under torch.no_grad() to obtain fixed cluster assignments r[n,k]
and fixed cluster means mu[k]. The diversity loss is then computed
differentiably using only the final weighted covariance step, so gradients
flow through seq → (seq - mu_fixed) → Sigma → log|Sigma|.

This avoids differentiating through unrolled EM iterations, giving a clean
gradient signal: push each sample away from its fixed cluster center.
"""

import torch


def _fit_gmm(seq, K=2, n_iters=5):
    """
    Fit a K-component GMM via EM (no gradient), then compute a differentiable
    diversity loss using fixed cluster assignments.

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

    # ── EM under no_grad — get fixed assignments ───────────────────────────────
    with torch.no_grad():
        # Initialise mu by splitting samples evenly across components
        chunk = N // K
        mu = torch.stack(
            [seq[:, :, k * chunk:(k + 1) * chunk].mean(dim=2) for k in range(K)],
            dim=2,
        )  # [H, B, K, A]

        # Initialise Sigma as global empirical covariance
        global_mean = seq.mean(dim=2, keepdim=True)
        d0    = seq - global_mean
        Sigma = (d0.unsqueeze(-1) * d0.unsqueeze(-2)).mean(dim=2)
        Sigma = Sigma.unsqueeze(2).expand(H, B, K, A, A).clone()

        pi = seq.new_full((H, B, K), 1.0 / K)

        TWO_PI = torch.tensor(2 * torch.pi, device=dev)

        for _ in range(n_iters):
            # E-step
            diff      = seq.unsqueeze(3) - mu.unsqueeze(2)          # [H, B, N, K, A]
            Sigma_reg = Sigma + reg
            L         = torch.linalg.cholesky(Sigma_reg)
            log_det   = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # [H, B, K]

            L_exp    = L.unsqueeze(2).expand(H, B, N, K, A, A)
            diff_col = diff.unsqueeze(-1)
            v        = torch.linalg.solve_triangular(L_exp, diff_col, upper=False)
            mahal    = (v * v).sum(dim=-2).squeeze(-1)               # [H, B, N, K]

            log_lik = -0.5 * (A * TWO_PI.log() + log_det.unsqueeze(2) + mahal)
            log_r   = log_lik + (pi + 1e-8).log().unsqueeze(2)
            r       = torch.softmax(log_r, dim=-1)                   # [H, B, N, K]

            # M-step
            N_k = r.sum(dim=2).clamp(min=1e-8)                      # [H, B, K]
            pi  = N_k / N
            mu  = (r.unsqueeze(-1) * seq.unsqueeze(3)).sum(dim=2) \
                  / N_k.unsqueeze(-1)

            diff  = seq.unsqueeze(3) - mu.unsqueeze(2)
            r_exp = r.unsqueeze(-1).unsqueeze(-1)
            Sigma = (r_exp * diff.unsqueeze(-1) * diff.unsqueeze(-2)).sum(dim=2) \
                    / N_k.unsqueeze(-1).unsqueeze(-1)

        # Fixed quantities — all detached
        r_fixed   = r
        pi_fixed  = pi
        mu_fixed  = mu
        N_k_fixed = N_k

    # ── Differentiable covariance with fixed assignments ───────────────────────
    # Gradient flows: seq → (seq - mu_fixed) → Sigma_diff → log|Sigma_diff|
    diff       = seq.unsqueeze(3) - mu_fixed.unsqueeze(2)            # [H, B, N, K, A]
    r_exp      = r_fixed.unsqueeze(-1).unsqueeze(-1)                 # [H, B, N, K, 1, 1]
    Sigma_diff = (r_exp * diff.unsqueeze(-1) * diff.unsqueeze(-2)).sum(dim=2) \
                 / N_k_fixed.unsqueeze(-1).unsqueeze(-1)             # [H, B, K, A, A]

    log_dets = torch.linalg.slogdet(Sigma_diff + reg)[1]            # [H, B, K]
    loss     = -(pi_fixed * log_dets).sum(dim=-1).mean()            # scalar

    with torch.no_grad():
        metrics = {
            'gmm_diversity':  -loss.item(),
            'gmm_pi_balance': pi_fixed.min(dim=-1).values.mean().item(),
        }
        for k in range(K):
            metrics[f'gmm_mu_{k}_norm']     = mu_fixed[:, :, k].norm(dim=-1).mean().item()
            metrics[f'gmm_sigma_{k}_trace'] = Sigma_diff[:, :, k].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
        if K == 2:
            metrics['gmm_inter_spread'] = \
                (mu_fixed[:, :, 0] - mu_fixed[:, :, 1]).norm(dim=-1).mean().item()

    return loss, metrics
