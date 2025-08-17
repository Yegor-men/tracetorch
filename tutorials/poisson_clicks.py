import torch, tracetorch
from typing import Optional, Tuple


def make_poisson_click_train(
		fps: int,
		listen_seconds: int,
		silence_seconds: int,
		retrieval_seconds: int = 1,
		n_left: int = 2,
		n_right: int = 2,
		n_retrieval: int = 1,
		p_left: float = 0.6,
		allow_overlap: bool = False,
		noise_prob: float = 0.02,  # baseline noise probability
		signal_prob: float = 0.5,  # additive probability when a click exists
		counts_left: Optional[int] = None,
		counts_right: Optional[int] = None,
		device: Optional[torch.device] = None,
		seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
	"""
	Returns (probs, spikes, label, left_more)
	  - probs: [T, C] float probabilities used to sample spikes (retrieval channel deterministic)
	  - spikes: [T, C] binary {0,1} spike matrix
	  - label: tensor([left_frac, right_frac]) fractions of listening-stage click frames
	  - left_more: bool, True if left clicks > right clicks

	Behavior:
	  - baseline noise_prob is present on all non-retrieval channels at all times
	  - where a listening-stage left/right click exists, add signal_prob to those channel probs
	  - retrieval channels are forced: 0 during listen+silence, 1 during recall (no sampling)
	"""
	if seed is not None:
		torch.manual_seed(seed)

	device = torch.device('cpu') if device is None else device

	frames_listen = int(listen_seconds * fps)
	frames_silence = int(silence_seconds * fps)
	frames_recall = int(retrieval_seconds * fps)
	T = frames_listen + frames_silence + frames_recall
	C = n_left + n_right + n_retrieval
	non_retrieval_C = n_left + n_right

	# --- Build listening-stage label vectors (0/1 per listening frame) ---
	if counts_left is not None or counts_right is not None:
		left_vec = torch.zeros(frames_listen, dtype=torch.float32, device=device)
		right_vec = torch.zeros(frames_listen, dtype=torch.float32, device=device)
		if counts_left is not None:
			assert 0 <= counts_left <= frames_listen
			idx_left = torch.randperm(frames_listen, device=device)[:counts_left]
			left_vec[idx_left] = 1.0
		if counts_right is not None:
			assert 0 <= counts_right <= frames_listen
			if allow_overlap:
				idx_right = torch.randperm(frames_listen, device=device)[:counts_right]
			else:
				avail = (~(left_vec.bool())).nonzero(as_tuple=False).view(-1)
				if counts_right > len(avail):
					raise ValueError("Not enough free frames for non-overlapping right clicks.")
				perm = torch.randperm(len(avail), device=device)[:counts_right]
				idx_right = avail[perm]
			right_vec[idx_right] = 1.0
	else:
		if allow_overlap:
			left_vec = (torch.rand(frames_listen, device=device) < p_left).float()
			right_vec = (torch.rand(frames_listen, device=device) < (1.0 - p_left)).float()
		else:
			# each listening frame gets exactly one choice: left with prob p_left, else right
			r = torch.rand(frames_listen, device=device)
			left_vec = (r < p_left).float()
			right_vec = ((r >= p_left) & (r < 1.0)).float()

	# compute ground-truth counts / fractions (based on listening-stage label vectors)
	left_count = float(left_vec.sum().item())
	right_count = float(right_vec.sum().item())
	if (left_count + right_count) == 0.0:
		label = torch.tensor([0.5, 0.5], dtype=torch.float32)
		left_more = False
	else:
		left_frac = left_count / (left_count + right_count)
		right_frac = right_count / (left_count + right_count)
		label = torch.tensor([left_frac, right_frac], dtype=torch.float32)
		left_more = left_count > right_count

	# --- Expand listening vectors across neurons (blocks) ---
	ones_left = torch.ones(n_left, device=device, dtype=torch.float32)
	ones_right = torch.ones(n_right, device=device, dtype=torch.float32)
	left_block = torch.einsum('t,k->tk', left_vec, ones_left)  # [frames_listen, n_left]
	right_block = torch.einsum('t,k->tk', right_vec, ones_right)  # [frames_listen, n_right]

	# silence & recall blocks for non-retrieval channels
	silence_left = torch.zeros((frames_silence, n_left), device=device)
	silence_right = torch.zeros((frames_silence, n_right), device=device)
	recall_left = torch.zeros((frames_recall, n_left), device=device)
	recall_right = torch.zeros((frames_recall, n_right), device=device)

	# retrieval neuron block (deterministic)
	recall_neurons = torch.ones((frames_recall, n_retrieval), device=device, dtype=torch.float32)

	# assemble 0/1 clean structure matrix probs_clean (0/1 signals for left/right and retrieval)
	listen_stage = torch.cat([left_block, right_block, torch.zeros((frames_listen, n_retrieval), device=device)], dim=1)
	silence_stage = torch.cat([silence_left, silence_right, torch.zeros((frames_silence, n_retrieval), device=device)],
							  dim=1)
	recall_stage = torch.cat([recall_left, recall_right, recall_neurons], dim=1)
	probs_clean = torch.cat([listen_stage, silence_stage, recall_stage], dim=0)  # [T, C]

	# --- Build actual probabilities ---
	probs = torch.zeros((T, C), device=device, dtype=torch.float32)

	# base noise on non-retrieval channels
	if non_retrieval_C > 0:
		base_noise = torch.full((T, non_retrieval_C), noise_prob, device=device, dtype=torch.float32)
		# add signal_prob where probs_clean indicates a click (listen-stage)
		signal_add = probs_clean[:, :non_retrieval_C] * signal_prob
		probs[:, :non_retrieval_C] = (base_noise + signal_add).clamp(0.0, 1.0)

	# retrieval channels deterministic (0 before recall; 1 during recall)
	recall_start = frames_listen + frames_silence
	recall_end = recall_start + frames_recall
	if n_retrieval > 0:
		probs[:, non_retrieval_C:] = 0.0
		probs[recall_start:recall_end, non_retrieval_C:] = 1.0

	# --- Sample spikes for all non-retrieval channels; set retrieval deterministically ---
	spikes = torch.zeros_like(probs)
	if non_retrieval_C > 0:
		spikes[:, :non_retrieval_C] = torch.bernoulli(probs[:, :non_retrieval_C])
	if n_retrieval > 0:
		spikes[recall_start:recall_end, non_retrieval_C:] = 1.0

	return probs, spikes, label, left_more


probs, spikes, label, left_more = make_poisson_click_train(
	fps=20,
	listen_seconds=5,
	silence_seconds=4,
	retrieval_seconds=1,
	n_left=10,
	n_right=10,
	n_retrieval=1,
	p_left=0.5,
	allow_overlap=False,
	noise_prob=0.05,
	signal_prob=0.9,
	seed=42
)

spike_train = []
for spike in spikes:
	spike_train.append(spike)

prob_train = []
for prob in probs:
	prob_train.append(prob)

tracetorch.plot.spike_train(spike_train, title=f"L/R Distribution: {label}")
tracetorch.plot.spike_train(prob_train, title=f"L/R Distribution: {label}")
