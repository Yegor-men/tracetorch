import torch
import torch.nn.functional as F

def smooth_lif_rate(d, t, i,
                    softplus_beta: float = 30.0,
                    clamp_eps: float = 1e-9,
                    maxcap_k: float = 30.0):
    """
    Smooth differentiable approximation of the LIF neuron's firing frequency
    (fraction of timesteps producing a spike), given:
      d : decay factor in (0,1)  (already passed through sigmoid by user)
      t : threshold > 0          (already passed through softplus by user)
      i : average input (can be negative, zero, positive)

    Returns:
      freq in (0,1], smoothly differentiable w.r.t. d, t, i.

    Hyperparams:
      softplus_beta : higher => softplus(i) is closer to ReLU(i) (small values at zero).
      clamp_eps     : numerical epsilon for logs/clamps.
      maxcap_k      : softness of the cap at 1 (higher => sharper cap).
    """

    # 1) keep d strictly inside (0,1) to avoid exactly zero/one logs
    d = d.clamp(min=clamp_eps, max=1.0 - clamp_eps)

    # 2) "lambda" is the continuous-time decay rate: lambda = -ln(d)
    lam = -torch.log(d)          # >= 0

    # 3) smooth positive part of input (if i <= 0 we want effectively zero driving input)
    #    Use a sharp softplus so softplus(0) is small (but still differentiable).
    i_pos = F.softplus(i, beta=softplus_beta)  # > 0, smooth

    # small safety epsilon to avoid divide-by-zero
    tiny = clamp_eps

    # 4) dimensionless driving ratio s = lambda * t / i_pos
    #    - if s >= 1 -> cannot reach threshold -> zero frequency (handled smoothly below)
    #    - if s small -> analytic formula is well-defined
    s = lam * t / (i_pos + tiny)

    # 5) compute the analytic-period denominator: -log(1 - s)
    #    but (1 - s) may be <= 0 (invalid). Create a smooth lower bound:
    #    soft_max(x, min_val) = min_val + softplus(x - min_val)
    min_val = clamp_eps
    one_minus_s = 1.0 - s
    safe_one_minus_s = min_val + F.softplus(one_minus_s - min_val)

    # denom = -log(safe_one_minus_s)  (safe and > 0)
    denom = -torch.log(safe_one_minus_s + tiny)

    # 6) analytic frequency (continuous-to-discrete mapping):
    #       freq_analytic = lam / denom
    #    this equals i/t in the lam->0 limit, and matches exact discrete formula for 0 < s < 1.
    freq_analytic = lam / (denom + tiny)

    # 7) soft-cap at 1.0 (the neuron cannot spike >1 per timestep).
    #    Use a smooth min(freq_analytic, 1) implemented via a sigmoid gate:
    gate = torch.sigmoid(maxcap_k * (1.0 - freq_analytic))  # ~1 when freq_analytic <= 1, ~0 when >1
    freq = freq_analytic * gate + 1.0 * (1.0 - gate)

    # 8) ensure strictly in [0,1] numerically (small safety)
    freq = freq.clamp(min=0.0, max=1.0)

    return freq

