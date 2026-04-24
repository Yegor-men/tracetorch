import torch
from torch import nn
from ..core import Layer


class SNG(Layer):
    """
    Spatial Neural Graph

    Nodes are primarily routed based on Euclidean distance between coordinates, a tensor of size [num_nodes, num_dimensions]

    However, the effective distance is obtained by dividing the Euclidean distance by the ratio of the flow values: a high
    ratio will mean that the proposed node is significantly better in flow, and hence significantly warps space.

    High flow nodes are used as outputs, low flow nodes are used as inputs.

    Inputs are injected additively, acting as sensory dendrites that stack on top of the network's recurrent predictive feedback.
    The output of any node is a residual addition: arcsinh(input) + delta created by the neurons (must be set to dim=-1)
    """

    def __init__(
            self,
            neurons,
            coordinates: torch.Tensor,
            flow_values: torch.Tensor,
            out_degrees: torch.Tensor,
            in_features: int,
            out_features: int,
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
        D.fill_diagonal_(float('inf'))  # Prevent self-connections
        # (We no longer mask input_idx, allowing the network to send predictive feedback to input nodes)

        # All nodes can route forward
        valid_src_indices = torch.arange(num_neurons)

        # 3) Sparse routing
        src_list, dst_list = [], []

        k_per_src = out_degrees.clone().long().to(coordinates.device)
        # Max connections is now num_neurons - 1 (since we only block self-connections)
        k_per_src = torch.clamp(k_per_src, min=1, max=num_neurons - 1)

        for i in valid_src_indices:
            real_k = k_per_src[i].item()
            best_dst = torch.topk(D[i], real_k, largest=False).indices
            src_list.append(torch.full((real_k,), i.item(), dtype=torch.long))
            dst_list.append(best_dst)

        src_tensor = torch.cat(src_list)
        dst_tensor = torch.cat(dst_list)
        self.register_buffer("src_indices", src_tensor)
        self.register_buffer("dst_indices", dst_tensor)

        # 4) Edge weights
        k_for_std = k_per_src.float()
        std_dev = 1.0 / torch.sqrt(torch.max(k_for_std, torch.tensor(1.0)))
        std_dev_per_edge = std_dev[src_tensor]
        self.edge_weights = nn.Parameter(torch.randn(len(src_tensor)) * std_dev_per_edge)

        # 5) Reservoir core
        self.neurons = neurons

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

        # Inject pure external inputs additively
        cumsum[..., self.input_idx] = cumsum[..., self.input_idx] + x_working

        # The reservoir computes the complex non-linear ODE based on the total accumulated current
        delta = self.neurons(cumsum)

        # The instantaneous state is a smoothened sum + the non-linear SNN spike
        new_trace = delta

        # Update the instantaneous state
        self.trace = self._from_working_dim(new_trace)

        # Output only the designated output nodes
        out_working = new_trace[..., self.output_idx]
        return self._from_working_dim(out_working)
