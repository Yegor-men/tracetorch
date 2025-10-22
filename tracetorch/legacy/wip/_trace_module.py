import torch
from torch import nn
from typing import List


class TraceModule(nn.Module):
	def __init__(self):
		super().__init__()
		self._update_mode = "immediate"

	def set_update_mode(self, mode: str, preserve_elig: bool = True):
		assert mode in {"immediate", "eligibility"}
		if self._update_mode == mode:
			return
		if mode == "eligibility":
			self.create_elig()
		else:
			if not preserve_elig:
				self.delete_elig()
		self._update_mode = mode

	@torch.no_grad()
	def create_elig(self):
		for name, p in self.named_parameters(recurse=False):
			elig_name = f"{name}_elig"
			if elig_name in self._buffers and self._buffers[elig_name] is not None:
				continue
			t = torch.zeros_like(p, device=p.device, dtype=p.dtype)
			self.register_buffer(elig_name, t, persistent=False)

	@torch.no_grad()
	def zero_elig(self):
		for key, buf in list(self._buffers.items()):
			if key.endswith("_elig") and buf is not None:
				buf.zero_()

	@torch.no_grad()
	def delete_elig(self):
		to_delete: List[str] = [k for k in list(self._buffers.keys()) if k.endswith("_elig")]
		for key in to_delete:
			del self._buffers[key]
			if hasattr(self, key):
				try:
					delattr(self, key)
				except Exception:
					setattr(self, key, None)

	@torch.no_grad()
	def _ensure_elig_created(self):
		if any(name.endswith("_elig") for name in self._buffers.keys()):
			return
		self.create_elig()

	@torch.no_grad()
	def elig_to_grad(self, scalar: float = 1.0, delete_after: bool = True):
		def accum_grad(param, g_raw):
			if param.grad is None:
				param.grad = g_raw.clone()
			else:
				param.grad.add_(g_raw)

		for name, param in self.named_parameters(recurse=False):
			elig_name = f"{name}_elig"
			elig = self._buffers.get(elig_name, None)
			if elig is None:
				continue
			g_raw = self.convert_elig_to_raw_grad(name, param, elig, scalar)
			accum_grad(param, g_raw)

		if delete_after:
			self.delete_elig()
		else:
			self.zero_elig()

	def convert_elig_to_raw_grad(
			self,
			name: str,
			param: torch.nn.Parameter,
			elig_tensor: torch.Tensor,
			scalar: float
	) -> torch.Tensor:
		return scalar * elig_tensor
