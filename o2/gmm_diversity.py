"""
gmm_diversity.py
----------------
Monte-Carlo cross-entropy diversity loss for DCEM.

EM is run under torch.no_grad() to fit a K-component GMM (frozen mu, Sigma,
pi). The diversity loss then evaluates the *mixture* log-density at each
(still grad-attached) sample under those frozen parameters and minimises
the average — i.e. minimises a Monte-Carlo estimate of cross-entropy
H(q, p) = -E_{x~q}[log p(x)], p = frozen GMM, q = sample distribution.

Because log p(x) = logsumexp_k(log pi_k + log N(x; mu_k, Sigma_k)) combines
all K components, the gradient on each sample is a soft blend over every
nearby mode (weighted by that mode's responsibility for the point), not
just its own assigned cluster — so it reflects inter-cluster overlap,
unlike a per-component log-det sum.

To revert to the previous detached-EM log-det diversity loss (intra-cluster
spread only, no cross-mode awareness): `git checkout -- o2/gmm_diversity.py`
restores it, since this MC cross-entropy version is uncommitted on top of
that original (commit 8aa4202).
"""

import torch


def _fit_gmm(seq, K=2, n_iters=5):
    """
    Fit a K-component GMM via detached EM, then estimate sample entropy via
    Monte-Carlo cross-entropy under the frozen mixture density.

    Args:
        seq:     [H, B, N, A] decoded action samples from the last CEM iteration.
        K:       number of GMM components.
        n_iters: number of EM iterations.

    Returns:
        loss:    scalar — mean log-density of samples under the frozen GMM
                 (minimise = push samples to lower-density regions = maximise entropy).
        metrics: dict of non-differentiable monitoring scalars.
    """
    H, B, N, A = seq.shape
    dev = seq.device
    reg = 1e-4 * torch.eye(A, device=dev)
    TWO_PI = torch.tensor(2 * torch.pi, device=dev)

    # ── EM under no_grad — fit frozen mu, Sigma, pi ─────────────────────────────
    with torch.no_grad():
        chunk = N // K
        mu = torch.stack(
            [seq[:, :, k * chunk:(k + 1) * chunk].mean(dim=2) for k in range(K)],
            dim=2,
        )  # [H, B, K, A]

        global_mean = seq.mean(dim=2, keepdim=True)
        d0    = seq - global_mean
        Sigma = (d0.unsqueeze(-1) * d0.unsqueeze(-2)).mean(dim=2)
        Sigma = Sigma.unsqueeze(2).expand(H, B, K, A, A).clone()

        pi = seq.new_full((H, B, K), 1.0 / K)

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

        mu_fixed    = mu
        pi_fixed    = pi
        Sigma_fixed = Sigma + reg

    # ── Differentiable Monte-Carlo cross-entropy under the frozen mixture ──────
    # Gradient flows: seq → diff → mahal → log_lik → logsumexp → loss
    L_fixed     = torch.linalg.cholesky(Sigma_fixed)                     # constant (no grad needed)
    log_det_fix = 2 * L_fixed.diagonal(dim1=-2, dim2=-1).log().sum(-1)   # [H, B, K]

    diff      = seq.unsqueeze(3) - mu_fixed.unsqueeze(2)                 # [H, B, N, K, A]
    L_fix_exp = L_fixed.unsqueeze(2).expand(H, B, N, K, A, A)
    v         = torch.linalg.solve_triangular(L_fix_exp, diff.unsqueeze(-1), upper=False)
    mahal     = (v * v).sum(dim=-2).squeeze(-1)                         # [H, B, N, K]

    log_lik = -0.5 * (A * TWO_PI.log() + log_det_fix.unsqueeze(2) + mahal)  # [H, B, N, K]
    log_p   = torch.logsumexp(log_lik + (pi_fixed + 1e-8).log().unsqueeze(2), dim=-1)  # [H, B, N]
    loss = log_p.mean()  # scalar — minimise to push samples toward lower density (higher entropy)

    with torch.no_grad():
        metrics = {
            'gmm_diversity':  -loss.item(),
            'gmm_pi_balance': pi_fixed.min(dim=-1).values.mean().item(),
        }
        for k in range(K):
            metrics[f'gmm_mu_{k}_norm']     = mu_fixed[:, :, k].norm(dim=-1).mean().item()
            metrics[f'gmm_sigma_{k}_trace'] = Sigma_fixed[:, :, k].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
        if K == 2:
            metrics['gmm_inter_spread'] = \
                (mu_fixed[:, :, 0] - mu_fixed[:, :, 1]).norm(dim=-1).mean().item()

    return loss, metrics
