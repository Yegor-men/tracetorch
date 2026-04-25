import torch
from torch import nn
from ..core import Layer


class TopologicalNeuralGraph(Layer):
    """
    Topological Neural Graph

    A structurally generated reservoir biased by exponential preferential attachment.
    Big nodes connect to big nodes. Inputs act as pure broadcasters, outputs act as deep integrators.
    """

    def __init__(
            self,
            neurons,
            in_features: int,
            out_features: int,
            avg_out_degree: float,
            dim: int = -1,
            temperature: float = None
    ):
        super().__init__(num_neurons=neurons.num_neurons, dim=dim)
        self.in_features = in_features
        self.out_features = out_features

        # Default the softmax temperature to avg_out_degree.
        # Mathematically, this perfectly stabilizes the Exponential distribution for Softmax.
        if temperature is None:
            temperature = torch.log1p(torch.tensor(avg_out_degree))

        # 1) Axons: Exponentially distributed out-degrees
        dist = torch.distributions.Exponential(1.0 / avg_out_degree)
        out_degrees = torch.round(dist.sample((self.num_neurons,))).long()

        # Clamp to ensure we never ask a neuron to connect to more neurons than actually exist
        out_degrees = torch.clamp(out_degrees, min=1, max=self.num_neurons - 1)

        # 2) Dendrites: Sparse Preferential Attachment
        src_list, dst_list = [], []

        # Scale out_degrees by temperature for numerical stability
        logits = torch.log1p(out_degrees.float()) / temperature

        for i in range(self.num_neurons):
            k = out_degrees[i].item()

            # Mask self out to prevent autapses (self-connections)
            curr_logits = logits.clone()
            curr_logits[i] = float('-inf')

            # Softmax to get probabilities (higher axon count = higher probability of being chosen)
            probs = torch.softmax(curr_logits, dim=0)

            # Sample best destinations without replacement
            best_dst = torch.multinomial(probs, k, replacement=False)

            src_list.append(torch.full((k,), i, dtype=torch.long))
            dst_list.append(best_dst)

        src_tensor = torch.cat(src_list)
        dst_tensor = torch.cat(dst_list)
        self.register_buffer("src_indices", src_tensor)
        self.register_buffer("dst_indices", dst_tensor)

        # 3) Edge weights (Identical variance scaling to FDSR)
        k_for_std = out_degrees.float()
        std_dev = 1.0 / torch.sqrt(torch.max(k_for_std, torch.tensor(1.0)))
        std_dev_per_edge = std_dev[src_tensor]
        self.edge_weights = nn.Parameter(torch.randn(len(src_tensor)) * std_dev_per_edge)

        # 4) Calculate actual in-degrees (dendrites) based on how the multinomial sorted it out
        in_degrees = torch.zeros(self.num_neurons, dtype=torch.float)
        in_degrees.scatter_add_(0, dst_tensor, torch.ones_like(dst_tensor, dtype=torch.float))

        # 5) Smart Input/Output Node Selection
        # Inputs: Broadcast well (High Axons), avoid recurrent interference (Low Dendrites)
        input_scores = out_degrees.float() / (in_degrees + 1.0)

        # Outputs: Gather diverse features (High Dendrites), avoid cyclic echoing (Low Axons)
        output_scores = in_degrees / (out_degrees.float() + 1.0)

        # Select Input nodes
        sorted_input_idx = torch.argsort(input_scores, descending=True)
        input_idx = sorted_input_idx[:in_features]
        self.register_buffer("input_idx", input_idx)

        # Select Output nodes (ensure they are mutually exclusive from inputs)
        output_scores_masked = output_scores.clone()
        output_scores_masked[input_idx] = -1.0
        sorted_output_idx = torch.argsort(output_scores_masked, descending=True)
        output_idx = sorted_output_idx[:out_features]
        self.register_buffer("output_idx", output_idx)

        # 6) Reservoir core
        self.neurons = neurons

        # 7) Instantaneous Spatial State
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

        # Inject pure external inputs
        # (Since we just strictly overwrite the index values, we inherently block incoming recurrent signals to inputs)
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
