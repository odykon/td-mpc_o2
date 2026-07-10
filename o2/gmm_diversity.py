"""
gmm_diversity.py
----------------
Monte-Carlo cross-entropy diversity loss for DCEM.

A K-component GMM (mu, Sigma, pi) is fit on a batch of fresh samples
(`_fit_gmm_em`). The E/M iterations that converge the cluster assignment
(responsibilities `r`) run under torch.no_grad() — differentiating through
the softmax/Cholesky machinery of a multi-iteration EM loop is unstable
(see CAUTION below) — but the *final* M-step is redone once more with `r`
frozen at its converged value and `seq` left attached, so mu/Sigma stay in
the autograd graph. This is justified by the envelope theorem: at a
converged fixed point, `r` is already optimal given (mu, Sigma, pi), so its
first-order sensitivity to a small perturbation of `seq` is zero — freezing
it and differentiating only through the (now linear/quadratic, well-
conditioned) M-step formula recovers the same first-order gradient as full
backprop through EM would, without the instability. Concretely: if `seq`
shifts uniformly (e.g. from a translation of u_mean), `mu` — a weighted
average with fixed weights — shifts by the same amount; if `seq` spreads
out more (u_std growing), the weighted second moment `Sigma` picks that up
directly. What it can't see is a sample switching which component it's
assigned to — an inherently discontinuous event that a smooth gradient
can't represent anyway.

The diversity penalty is then the mixture log-density evaluated at a
separate point (`_gmm_log_prob`) — typically the single sample actually
used for the value backward pass, SAC-style — minimised to push that point
toward the mixture's lower-density regions, i.e. to maximise entropy at the
point that matters. Because both the fit and the scored point trace back to
the same u_mean/u_std, a pure translation of u_mean moves the fit's mean
along with it, resisting the loss being satisfied by relocating u_mean
alone rather than growing spread.

Because log p(x) = logsumexp_k(log pi_k + log N(x; mu_k, Sigma_k)) combines
all K components, the gradient blends over every nearby mode (weighted by
that mode's responsibility for the point), not just its own assigned
cluster — so it reflects inter-cluster overlap, unlike a per-component
log-det sum.

CAUTION — fully differentiating through EM (i.e. not freezing `r`): the
E-step's softmax responsibilities saturate toward one-hot as EM converges
(vanishing gradient through later iterations), the M-step's
`N_k.clamp(min=1e-8)` has exactly zero gradient for a starved/collapsing
component, `log(pi + 1e-8)` explodes as any component's weight approaches
0, and `cholesky`/`solve_triangular` get ill-conditioned as a component's
covariance shrinks — all of this compounds over `n_iters` unrolled
iterations. This is why only the final M-step is left differentiable here.

REVERT to fully detached EM (frozen fit, no gradient through it at all):
wrap the whole body of `_fit_gmm_em` below in `with torch.no_grad():`
(including the final M-step redo), and in o2/decoder_updates.py restore
`.detach()` on `u_mean`/`u_std` when building `gmm_u`, and re-wrap the
fit-stage `decode_sequence` call in `torch.no_grad()`.

To revert further, to the original per-sample log-det diversity loss
(intra-cluster spread only, no cross-mode awareness, no fit/eval split):
`git checkout -- o2/gmm_diversity.py` restores it, since this MC
cross-entropy version is uncommitted on top of that original (commit
8aa4202).
"""

import torch


def _fit_gmm_em(seq, K=2, n_iters=5):
    """
    Fit a K-component GMM to seq via EM. The E/M loop that converges the
    responsibilities runs under no_grad (stable); the final M-step is then
    redone with those responsibilities frozen but seq attached, so mu/Sigma
    carry gradient back to seq without differentiating through the E-step
    itself. See module docstring for the envelope-theorem justification.

    Args:
        seq:     [H, B, N, A] fresh decoded action samples to fit the mixture to.
        K:       number of GMM components.
        n_iters: number of EM iterations.

    Returns:
        mu, Sigma: mixture mean/covariance, [H,B,K,A], [H,B,K,A,A] —
                   differentiable w.r.t. seq via the frozen-responsibility
                   final M-step.
        pi:        mixture weights, [H,B,K] — fully detached (depends only
                   on the frozen responsibilities, not on seq directly).
        metrics:   dict of non-differentiable fit-diagnostic scalars.
    """
    H, B, N, A = seq.shape
    dev = seq.device
    reg = 1e-4 * torch.eye(A, device=dev)
    TWO_PI = torch.tensor(2 * torch.pi, device=dev)

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

        r_fixed   = r      # converged responsibilities from the last E-step, frozen
        pi_fixed  = pi
        N_k_fixed = N_k

    # Final M-step redone with seq attached, r frozen at its converged
    # value — carries the first-order sensitivity of mu/Sigma to seq
    # (translation, spread) without differentiating through the E-step.
    mu_fixed = (r_fixed.unsqueeze(-1) * seq.unsqueeze(3)).sum(dim=2) / N_k_fixed.unsqueeze(-1)

    diff  = seq.unsqueeze(3) - mu_fixed.unsqueeze(2)
    r_exp = r_fixed.unsqueeze(-1).unsqueeze(-1)
    Sigma_fixed = (r_exp * diff.unsqueeze(-1) * diff.unsqueeze(-2)).sum(dim=2) \
                  / N_k_fixed.unsqueeze(-1).unsqueeze(-1)
    Sigma_fixed = Sigma_fixed + reg

    with torch.no_grad():
        metrics = {
            'gmm_pi_balance': pi_fixed.min(dim=-1).values.mean().item(),
        }
        for k in range(K):
            metrics[f'gmm_mu_{k}_norm']     = mu_fixed[:, :, k].norm(dim=-1).mean().item()
            metrics[f'gmm_sigma_{k}_trace'] = Sigma_fixed[:, :, k].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
        if K == 2:
            metrics['gmm_inter_spread'] = \
                (mu_fixed[:, :, 0] - mu_fixed[:, :, 1]).norm(dim=-1).mean().item()

    return mu_fixed, Sigma_fixed, pi_fixed, metrics


def _gmm_log_prob(x, mu, Sigma, pi):
    """
    Differentiable mixture log-density of x under a frozen GMM.

    Args:
        x:     [H, B, M, A] points to score — gradient flows through this.
        mu:    [H, B, K, A] frozen means (from `_fit_gmm_em`).
        Sigma: [H, B, K, A, A] frozen, already-regularised covariances.
        pi:    [H, B, K] frozen mixture weights.

    Returns:
        log_p: [H, B, M] differentiable log-density of each point.
    """
    H, B, M, A = x.shape
    K = mu.shape[2]
    TWO_PI = torch.tensor(2 * torch.pi, device=x.device)

    L       = torch.linalg.cholesky(Sigma)                       # constant (no grad needed)
    log_det = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)      # [H, B, K]

    diff  = x.unsqueeze(3) - mu.unsqueeze(2)                      # [H, B, M, K, A]
    L_exp = L.unsqueeze(2).expand(H, B, M, K, A, A)
    v     = torch.linalg.solve_triangular(L_exp, diff.unsqueeze(-1), upper=False)
    mahal = (v * v).sum(dim=-2).squeeze(-1)                       # [H, B, M, K]

    log_lik = -0.5 * (A * TWO_PI.log() + log_det.unsqueeze(2) + mahal)  # [H, B, M, K]
    log_p   = torch.logsumexp(log_lik + (pi + 1e-8).log().unsqueeze(2), dim=-1)  # [H, B, M]
    return log_p
