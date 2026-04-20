import torch
from torch import nn
from ._snnlayer import Layer as SNNLayer
from ._lib_layers import LIB
import math


class FDSR(SNNLayer):
    """
    Flow-Directed Spatial Reservoir (Ultra-Lean Version)

    - Nodes route signals to nearest neighbors, biased by 'flow' (distance to output).
    - Temporal memory is handled entirely by the SNN's ODE membrane physics.
    - Graph state updates instantly via: Trace = Average_Input + SNN_Spike(Total_Input).
    """

    def __init__(
            self,
            lif_neurons,
            coordinates: torch.Tensor,
            flow_values: torch.Tensor,
            in_features: int,
            out_features: int,
            num_connections: int,
            dim: int = -1,
    ):
        num_neurons, num_dims = coordinates.shape
        self.num_neurons = num_neurons
        self.num_dims = num_dims
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(num_neurons=num_neurons, dim=dim)

        sorted_indices = torch.argsort(flow_values)
        self.register_buffer("input_idx", sorted_indices[:in_features])
        self.register_buffer("output_idx", sorted_indices[-out_features:])

        # 1) Effective distance matrix (Euclidean / flow ratio)
        d = torch.cdist(coordinates, coordinates, p=2.0)
        r = flow_values.unsqueeze(0) / flow_values.unsqueeze(1)
        D = d / r

        # 2) Topology masking
        D.fill_diagonal_(float('inf'))
        D[:, self.input_idx] = float('inf')  # Inputs don't receive recurrent signals

        # All nodes can route forward
        valid_src_indices = torch.arange(num_neurons)

        # 3) Sparse routing
        src_list, dst_list = [], []
        k = min(num_connections, num_neurons - in_features - 1)

        for i in valid_src_indices:
            best_dst = torch.topk(D[i], k, largest=False).indices
            src_list.append(torch.full((k,), i.item(), dtype=torch.long))
            dst_list.append(best_dst)

        src_tensor = torch.cat(src_list)
        dst_tensor = torch.cat(dst_list)
        self.register_buffer("src_indices", src_tensor)
        self.register_buffer("dst_indices", dst_tensor)

        # Pre-calculate in-degrees for cumavg (critical for bounding magnitude)
        in_degrees = torch.zeros(num_neurons)
        unique_dst, counts = torch.unique(dst_tensor, return_counts=True)
        in_degrees[unique_dst] = counts.float()
        in_degrees[in_degrees == 0] = 1.0  # Prevent division by zero
        self.register_buffer("in_degrees", in_degrees)

        # 4) Edge weights
        std_dev = 1.0 / math.sqrt(max(1, k))
        self.edge_weights = nn.Parameter(torch.randn(len(src_tensor)) * std_dev)

        # 5) SNN Core
        self.lif = lif_neurons

        # 6) Instantaneous Spatial State
        self._initialize_state("trace")

    def forward(self, x):
        self._ensure_states(x)

        x_working = self._to_working_dim(x)
        trace_working = self._to_working_dim(self.trace)

        # Route previous trace through graph to get Total Current (cumsum)
        src_acts = trace_working[..., self.src_indices]
        weighted_acts = src_acts * self.edge_weights

        cumsum = torch.zeros_like(trace_working)
        expanded_dst = self.dst_indices.expand(*trace_working.shape[:-1], -1)
        cumsum.scatter_add_(-1, expanded_dst, weighted_acts)

        # Calculate Average Excitement (cumavg) to prevent infinite loops from exploding
        cumavg = cumsum / self.in_degrees

        # Inject pure external inputs
        cumsum[..., self.input_idx] = x_working
        cumavg[..., self.input_idx] = x_working

        # The SNN computes the complex non-linear ODE based on the total accumulated current
        delta = self.lif(cumsum)

        # The instantaneous state is simply the baseline average + the non-linear SNN spike
        new_trace = cumavg + delta

        # Update the instantaneous state
        self.trace = self._from_working_dim(new_trace)

        # Output only the designated output nodes
        return new_trace[..., self.output_idx]
