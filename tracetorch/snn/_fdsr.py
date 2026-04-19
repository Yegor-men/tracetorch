import torch
from torch import nn
from ._snnlayer import Layer as SNNLayer
from ._lib_layers import LIB
import math
from typing import Union, Literal


class FDSR(SNNLayer):
    """
    Flow-Directed Spatial Reservoir

    Nodes (SNN neurons) get a random coordinate. Based on this coordinate, we know their "flow", how close they are to
    the output end of the FDSR module. We calculate this by projecting their coordinate on to a ones vector (equivalent to summing the coordinates among dimensions)
    and then we multiply this value by the actual flow argument, and then exponentiate to create the real flow value.

    Nodes connect primarily due to Euclidean distance, but the presence of flow makes them more biased towards pointing one way: toward the exit.
    this is because between node A and B, we can calculate the ratio, if it's larger than 1, then that means that B is closer to the exist than A and vice versa.
    we divide the actual distance by the ratio to hence create the effective distance and thus the neurons have a chance to connect backwards

    Gamma and a trace are used to record a neuron's outputs so that they can be used in the next timestep, but also
    because it's literally the previous layer at the same time, residual addition is made, but instead of cumsum, the cumavg isused instead for stabiltiy.

    Mathematically, it works out, the trace is updated as such (t stands for trace, a stands for avg, D stands for delta, d stands for decay):
    t = td + (a + D) * (1 - d)
    td - t + (a + D) * (1 - d) = 0
    t(d - 1) = -(a + D) * (1 - d)
    t(1 - d) = (a + D) * (1 - d)
    t = a + D

    so thus the trace will stabilize at avg + delta. All good :) I think :( (?)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_neurons: int,
            num_connections: int,
            gamma: Union[float, torch.Tensor] = 0.9,
            spk_scale: Union[float, torch.Tensor] = 0.0,
            num_dims: int = 3,
            flow: float = 0.1,
            dim: int = -1,
            gamma_rank: Literal[0, 1] = 1,
            spk_scale_rank: Literal[0, 1] = 1,
            learn_gamma: bool = True,
            learn_spk_scale: bool = True,
    ):
        super().__init__(num_neurons=num_neurons, dim=dim)

        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons

        # 1) spatial embedding and topological flow
        coords = torch.randn(num_neurons, num_dims)
        flow_vals = torch.exp(flow * coords.sum(dim=1))

        sorted_indices = torch.argsort(flow_vals)
        self.register_buffer("input_idx", sorted_indices[:in_features])
        self.register_buffer("output_idx", sorted_indices[-out_features:])

        # 2) effective distance matrix (euclidean / flow ratio)
        d = torch.cdist(coords, coords, p=2.0)
        r = flow_vals.unsqueeze(0) / flow_vals.unsqueeze(1)
        D = d / r

        # 3) topology masking so that input neurons don't listen to FSDR and output neurons don't feed FSDR
        D.fill_diagonal_(float('inf'))
        D[:, self.input_idx] = float('inf')  # inputs don't receive recurrent signals

        is_valid_src = torch.ones(num_neurons, dtype=torch.bool)
        is_valid_src[self.output_idx] = False  # outputs don't send recurrent signals
        valid_src_indices = torch.nonzero(is_valid_src).squeeze(-1)

        # 4) sparse routing
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

        # pre-calculate the in degree of every neuron to calculate the cumav later (critical for stability in place of cumsum)
        in_degrees = torch.zeros(num_neurons)
        unique_dst, counts = torch.unique(dst_tensor, return_counts=True)
        in_degrees[unique_dst] = counts.float()
        in_degrees[in_degrees == 0] = 1.0  # Prevent division by zero for isolated/input nodes
        self.register_buffer("in_degrees", in_degrees)

        # edge weights connecting the nodes
        std_dev = 1.0 / math.sqrt(max(1, k))
        self.edge_weights = nn.Parameter(torch.randn(len(src_tensor)) * std_dev)

        # LIF acts as an ODE with the quant_fn set to nn.Identity()
        self.lif = LIB(
            num_neurons=num_neurons,
            beta=torch.rand(num_neurons),
            threshold=torch.rand(num_neurons),
            bias=torch.randn(num_neurons) * 0.1,
            quant_fn=nn.Identity(),
        )

        self._initialize_state("trace")

        # EMA decay used for recording the outputs, which is actually used as the inputs/previous layer outputs
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        # spk_scale is a zero initialized projection from SNN -> float, needed for stability and gradient flow since it's effectively residual
        self._register_parameter("spk_scale", spk_scale, spk_scale_rank, learn_spk_scale)

    def forward(self, x):
        self._ensure_states(x)

        x_working = self._to_working_dim(x)
        trace_working = self._to_working_dim(self.trace)

        # calculate recorded trace (previous "layer" into cumsum)
        src_acts = trace_working[..., self.src_indices]
        weighted_acts = src_acts * self.edge_weights

        cumsum = torch.zeros_like(trace_working)
        expanded_dst = self.dst_indices.expand(*trace_working.shape[:-1], -1)
        cumsum.scatter_add_(-1, expanded_dst, weighted_acts)

        # calculate average dendritic excitement (cumavg)
        cumavg = cumsum / self.in_degrees

        # inject actual external input into the neurons
        # inputs overwrite everything since they've no connection to recurrent layers
        cumsum[..., self.input_idx] = x_working
        cumavg[..., self.input_idx] = x_working

        # fire the SNN via the cumsum
        spikes_working = self.lif(cumsum)

        # create the delta
        delta = spikes_working * self.spk_scale

        # update the trace (previous layer's outputs) to create the effect of residual addition
        new_trace = trace_working * self.gamma + (cumavg + delta) * (1 - self.gamma)

        self.trace = self._from_working_dim(new_trace)

        # output only the designated output nodes
        return new_trace[..., self.output_idx]
