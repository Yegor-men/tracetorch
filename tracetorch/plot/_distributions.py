from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt


def _safe_to_numpy(t: torch.Tensor) -> np.ndarray:
	"""Flatten a tensor to finite CPU NumPy values for plotting."""
	t = t.detach().cpu().ravel()
	a = t.numpy()
	# remove nans and infs
	a = a[np.isfinite(a)]
	return a


def _silverman_bandwidth(x: np.ndarray) -> float:
	"""Estimate a Gaussian KDE bandwidth using Silverman's rule of thumb."""
	n = x.size
	if n < 2:
		return 1.0
	s = np.std(x, ddof=1)
	# Silverman's rule of thumb for Gaussian kernel
	bw = 1.06 * s * n ** (-1 / 5)
	# guard against zero bandwidth
	if bw <= 0 or not np.isfinite(bw):
		bw = np.ptp(x) / 10.0 if np.ptp(x) > 0 else 1.0
	return bw


def _gaussian_kde_eval(samples: np.ndarray,
					   grid: np.ndarray,
					   bw: Optional[float] = None,
					   max_samples: int = 20000) -> np.ndarray:
	"""Evaluate a Gaussian KDE over a grid.

	Args:
		samples (np.ndarray): one-dimensional sample array.
		grid (np.ndarray): one-dimensional evaluation grid.
		bw (float, optional): kernel bandwidth. If ``None``, Silverman's rule is
			used.
		max_samples (int, default=20000): maximum samples used for KDE; larger
			inputs are deterministically subsampled.

	Returns:
		np.ndarray: density values aligned with ``grid`` and normalized to
		integrate approximately to one.
	"""
	if samples.size == 0:
		return np.zeros_like(grid, dtype=float)
	x = samples
	n = x.size
	if n > max_samples:
		rng = np.random.default_rng(0)
		x = rng.choice(x, size=max_samples, replace=False)
		n = x.size
	if bw is None:
		bw = _silverman_bandwidth(x)
	# compute kernel densities: sum of Gaussians
	# density = (1/(n*bw)) * sum_j phi((grid - x_j)/bw)
	# vectorized with broadcasting (grid[:,None] - x[None,:])
	diffs = (grid[:, None] - x[None, :]) / bw
	# gaussian kernel
	K = np.exp(-0.5 * diffs * diffs) / np.sqrt(2 * np.pi)
	dens = K.sum(axis=1) / (n * bw)
	# normalize numerical integration to 1 (improves comparability across grids)
	area = np.trapz(dens, grid)
	if area > 0:
		dens = dens / area
	return dens


def distributions(layers: List[torch.Tensor],
				  title: str,
				  *,
				  n_grid: int = 1024,
				  n_percentiles: int = 100,
				  bandwidths: Optional[List[Optional[float]]] = None,
				  show_percentile_lines: bool = False,
				  compute_metrics: bool = True,
				  max_kde_samples: int = 20000
				  ) -> Dict[str, Any]:
	"""Estimate and plot value distributions for tensors.

	Args:
		layers (List[torch.Tensor]): tensors to flatten and compare.
		title (str): plot title.
		n_grid (int, default=1024): number of KDE evaluation points.
		n_percentiles (int, default=100): number of percentile buckets.
		bandwidths (List[Optional[float]], optional): per-layer KDE bandwidths.
			``None`` entries use Silverman's rule.
		show_percentile_lines (bool, default=False): if True, show selected
			percentile markers.
		compute_metrics (bool, default=True): if True, compute pairwise KDE L1
			distance and Wasserstein distance when SciPy is available.
		max_kde_samples (int, default=20000): maximum samples per tensor used for
			KDE evaluation.

	Returns:
		Dict[str, Any]: plotting data including ``grid``, ``kdes``,
		``percentiles``, ``stats``, ``fig_ax``, and optional ``metrics``.

	Notes:
		This helper is intended for exploratory analysis and calls ``plt.show()``.
	"""
	# prepare data arrays
	arrs = [_safe_to_numpy(t) for t in layers]
	stats = []
	# compute global x-range for consistent plotting
	finite_arrs = [a for a in arrs if a.size > 0]
	if len(finite_arrs) == 0:
		raise ValueError("No finite values found in any input tensor.")
	global_min = min(a.min() for a in finite_arrs)
	global_max = max(a.max() for a in finite_arrs)
	# expand a bit for display
	pad = 0.01 * (global_max - global_min) if global_max > global_min else 1.0
	x_min = global_min - pad
	x_max = global_max + pad
	grid = np.linspace(x_min, x_max, n_grid, dtype=float)

	if bandwidths is None:
		bandwidths = [None] * len(arrs)
	elif len(bandwidths) != len(arrs):
		raise ValueError("If provided, bandwidths must have same length as layers")

	kdes = []
	percentiles = []
	for i, (a, bw) in enumerate(zip(arrs, bandwidths)):
		if a.size == 0:
			kdes.append(np.zeros_like(grid))
			percentiles.append(np.full(n_percentiles + 1, np.nan))
			stats.append({'n': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan})
			continue
		dens = _gaussian_kde_eval(a, grid, bw=bw, max_samples=max_kde_samples)
		kdes.append(dens)
		pct_values = np.percentile(a, np.linspace(0, 100, n_percentiles + 1))
		percentiles.append(pct_values)
		stats.append({'n': int(a.size), 'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
					  'min': float(a.min()), 'max': float(a.max())})

	# plotting
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.set_title(title)
	ax.set_xlabel("value")
	ax.set_ylabel("density (KDE)")
	lines = []
	labels = []
	for i, dens in enumerate(kdes):
		# label contains index and base stats
		s = stats[i]
		label = f"layer {i} (n={s['n']}, μ={s['mean']:.3g}, σ={s['std']:.3g})"
		l, = ax.plot(grid, dens, label=label)
		lines.append(l)
		labels.append(label)
		if show_percentile_lines and np.isfinite(percentiles[i]).all():
			# show a few percentiles lightly (10th, 50th, 90th)
			for p in (10, 50, 90):
				v = np.percentile(arrs[i], p)
				ax.axvline(v, linestyle='--', linewidth=0.6, alpha=0.6)

	ax.legend(loc='best', fontsize='small')
	ax.grid(True, linestyle=':', alpha=0.4)
	plt.tight_layout()
	plt.show()

	result: Dict[str, Any] = {
		'grid': grid,
		'kdes': kdes,
		'percentiles': percentiles,
		'stats': stats,
		'fig_ax': (fig, ax)  # note: matplotlib objects included for programmatic use
	}

	# compute pairwise similarity metrics
	if compute_metrics:
		# try to use scipy for Wasserstein and KS if available; otherwise fall back to grid L1
		pairwise_wasserstein = None
		pairwise_l1 = np.zeros((len(arrs), len(arrs)), dtype=float)
		try:
			from scipy.stats import wasserstein_distance
			use_wasserstein = True
			pairwise_wasserstein = np.zeros((len(arrs), len(arrs)), dtype=float)
		except Exception:
			use_wasserstein = False
			pairwise_wasserstein = None

		# L1 between normalized KDEs on the grid: integral |f - g|
		for i in range(len(arrs)):
			for j in range(len(arrs)):
				f = kdes[i]
				g = kdes[j]
				pairwise_l1[i, j] = np.trapz(np.abs(f - g), grid)
				if use_wasserstein:
					# if one of the arrays empty, put nan
					a = arrs[i]
					b = arrs[j]
					if a.size == 0 or b.size == 0:
						pairwise_wasserstein[i, j] = np.nan
					else:
						pairwise_wasserstein[i, j] = wasserstein_distance(a, b)

		metrics = {'l1_kde': pairwise_l1}
		if use_wasserstein:
			metrics['wasserstein'] = pairwise_wasserstein
		result['metrics'] = metrics

	return result
