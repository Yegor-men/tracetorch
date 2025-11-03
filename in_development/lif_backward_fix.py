import torch
import torch.nn.functional as F


def smooth_lif_rate_vector(i, d, t,
						   beta: float = 10.0,
						   k1: float = 30.0,
						   k2: float = 60.0,
						   eps: float = 1e-8):
	"""
	Vectorized smooth LIF firing frequency approximation for tensors i, d, t (same shape).
	All operations are differentiable and auto-friendly.

	Args:
	  i: tensor of average inputs (shape [n])
	  d: tensor of decay factors in (0,1) (shape [n])
	  t: tensor of thresholds (>0 ideally) (shape [n])
	  beta: softplus stiffness (higher => closer to ReLU(i))
	  k1: "immediate-fire" gate sharpness (higher => sharper switch to f=1 when i >= t)
	  k2: "unreachable" gate sharpness (higher => sharper suppression when s -> 1)
	  eps: small numeric epsilon for stability

	Returns:
	  freq: tensor of shape [n] with values in [0,1] (smoothly differentiable)
	"""

	# ensure same device/dtype and numerical stability
	i = torch.as_tensor(i)
	d = torch.as_tensor(d)
	t = torch.as_tensor(t)

	# clamp d to (eps, 1-eps) to avoid log(0)
	d_clamped = d.clamp(min=eps, max=1.0 - eps)

	# soft positive input: i_pos = softplus(i, beta)
	# using torch.nn.functional.softplus with beta gives (1/beta) * ln(1 + exp(beta * x))
	i_pos = F.softplus(i, beta=beta)

	# lambda = -ln(d)
	lam = -torch.log(d_clamped)

	# s = t * (1 - d) / (i_pos + eps)
	s = (t * (1.0 - d_clamped)) / (i_pos + eps)

	# safe "1 - s" with a smooth lower bound using softplus:
	# safe_one_minus_s = eps + softplus( (1 - s) - eps )
	one_minus_s = 1.0 - s
	safe_one_minus_s = eps + F.softplus(one_minus_s - eps)

	# denom = -ln(safe_one_minus_s)  (guaranteed > 0)
	denom = -torch.log(safe_one_minus_s + eps)

	# raw analytic frequency (may be > 1 or small)
	raw = lam / (denom + eps)

	# immediate-fire gate: if i_pos >> t -> g_i ~ 1 -> freq ≈ 1
	g_i = torch.sigmoid(k1 * (i_pos - t))

	# reachability gate: suppress when s approaches 1 (we want g_s ~ 0 near s >= 1)
	# choose pivot close to 1 (e.g., 0.999) so we smoothly go to zero for s near 1
	pivot = 0.999
	g_s = torch.sigmoid(k2 * (pivot - s))  # ~1 when s << pivot, ~0 when s >> pivot

	# combine:
	# - if i_pos >> t -> g_i ~ 1 so freq ~ 1
	# - else use (g_s * raw) to smoothly go to raw in valid region, and to 0 when s >= 1
	freq = g_i * 1.0 + (1.0 - g_i) * (g_s * raw)

	# final safety: clamp into [0,1]
	freq = freq.clamp(min=0.0, max=1.0)

	return freq


n = round(1e8)
print(f"N: {n:,}")

for _ in range(100):
	rand_i = (torch.randn(n) * 0.5).to("cuda")
	rand_d = torch.rand(n).to("cuda")
	rand_t = torch.nn.functional.softplus(torch.randn(n)).to("cuda")

	frequencies = smooth_lif_rate_vector(rand_i, rand_d, rand_t)

	over_1 = int((frequencies > 1).float().sum().item())
	under_0 = int((frequencies < 0).float().sum().item())

	if over_1 != 0 or under_0 != 0:
		print(f"Under 0: {under_0} - {(under_0 / n) * 100}%")
		print(f"Over 1: {over_1} - {(over_1 / n) * 100}%")
	else:
		print(f"{n * (_ + 1):,} numbers passed")
