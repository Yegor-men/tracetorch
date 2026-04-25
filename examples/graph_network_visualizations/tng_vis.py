import torch
import numpy as np
import plotly.graph_objects as go

# ==========================================
# HYPERPARAMETERS TO PLAY WITH
# ==========================================
NUM_NEURONS = 4096  # Number of nodes in the graph
IN_FEATURES = 64  # Number of input nodes (Pure broadcasters)
OUT_FEATURES = 64  # Number of output nodes (Deep integrators)
AVG_OUT_DEGREE = 8.0  # Average number of axons per neuron


# ==========================================

def pytorch_spring_layout(src, dst, num_nodes, dim=3, iterations=150):
    src = torch.as_tensor(src, dtype=torch.long)
    dst = torch.as_tensor(dst, dtype=torch.long)

    # Initialize random positions
    pos = torch.rand(num_nodes, dim) * 2 - 1.0

    # Ideal distance between nodes
    k = 1.0 / (num_nodes ** (1.0 / dim))
    t = 1.0  # Initial "temperature"

    eye = torch.eye(num_nodes, dtype=torch.bool)

    for _ in range(iterations):
        # --- 1. Repulsion (All nodes push apart) ---
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, dim]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [N, N, 1]
        dist[eye] = 1.0  # Prevent divide-by-zero on diagonal

        repulsion = (k ** 2 / (dist ** 2 + 1e-9)) * diff
        repulsion[eye] = 0.0  # Remove self-repulsion
        disp = repulsion.sum(dim=1)  # [N, dim]

        # --- 2. Attraction (Connected nodes pull together) ---
        edge_diff = pos[dst] - pos[src]  # [E, dim]
        edge_dist = torch.norm(edge_diff, dim=-1, keepdim=True)
        attraction = (edge_dist / k) * edge_diff  # [E, dim]

        # Add forces to source, subtract from destination
        disp.scatter_add_(0, src.unsqueeze(1).expand(-1, dim), attraction)
        disp.scatter_add_(0, dst.unsqueeze(1).expand(-1, dim), -attraction)

        # --- 3. Update Positions ---
        disp_norm = torch.norm(disp, dim=-1, keepdim=True)
        disp_norm[disp_norm == 0] = 1e-9

        # Scale displacement by unit vector, capped by current temperature t
        disp = disp / disp_norm * torch.clamp(disp_norm, max=t)
        pos += disp

        # Cool down the temperature
        t *= 0.98

        # Center around origin and scale for visualization
    pos -= pos.mean(dim=0)
    max_std = pos.std(dim=0).max()
    if max_std > 0:
        pos /= max_std

    return pos.numpy()


def generate_sing_graph(num_neurons, in_features, out_features, avg_out_degree):
    """Executes the exact SING initialization logic"""
    torch.manual_seed(42)

    # 1. Axons: Exponentially distributed out-degrees
    dist = torch.distributions.Exponential(1.0 / avg_out_degree)
    out_degrees = torch.round(dist.sample((num_neurons,))).long()
    out_degrees = torch.clamp(out_degrees, min=1, max=num_neurons - 1)

    # 2. Dendrites: Sparse Preferential Attachment
    src_list, dst_list = [], []
    logits = torch.log1p(out_degrees) / torch.log1p(torch.tensor(avg_out_degree))

    for i in range(num_neurons):
        k = out_degrees[i].item()

        curr_logits = logits.clone()
        curr_logits[i] = float('-inf')  # Prevent self-connections

        probs = torch.softmax(curr_logits, dim=0)
        best_dst = torch.multinomial(probs, k, replacement=False)

        src_list.append(torch.full((k,), i, dtype=torch.long))
        dst_list.append(best_dst)

    src_tensor = torch.cat(src_list)
    dst_tensor = torch.cat(dst_list)

    # 3. Calculate actual in-degrees (Dendrites)
    in_degrees = torch.zeros(num_neurons, dtype=torch.float)
    in_degrees.scatter_add_(0, dst_tensor, torch.ones_like(dst_tensor, dtype=torch.float))

    # 4. Smart Input/Output Node Selection
    input_scores = out_degrees.float() / (in_degrees + 1.0)
    output_scores = in_degrees / (out_degrees.float() + 1.0)

    # Select Inputs
    sorted_input_idx = torch.argsort(input_scores, descending=True)
    input_idx = sorted_input_idx[:in_features]

    # Select Outputs (Mutually exclusive from inputs)
    output_scores_masked = output_scores.clone()
    output_scores_masked[input_idx] = -1.0
    sorted_output_idx = torch.argsort(output_scores_masked, descending=True)
    output_idx = sorted_output_idx[:out_features]

    return (
        src_tensor.numpy(), dst_tensor.numpy(),
        out_degrees.numpy(), in_degrees.numpy(),
        input_idx.numpy(), output_idx.numpy()
    )


def plot_sing():
    print(f"Generating purely topological SING network with {NUM_NEURONS} nodes...")
    src, dst, out_degs, in_degs, input_idx, output_idx = generate_sing_graph(
        NUM_NEURONS, IN_FEATURES, OUT_FEATURES, AVG_OUT_DEGREE
    )

    # ---------------------------------------------------------
    # LAYOUT: Use custom PyTorch physics solver!
    # ---------------------------------------------------------
    print("Simulating 3D physics layout (PyTorch Force-Directed)...")
    pos = pytorch_spring_layout(src, dst, NUM_NEURONS, dim=3)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    node_colors = [''] * NUM_NEURONS

    # Size nodes by their total connections (log-scaled for visibility)
    total_degrees = out_degs + in_degs
    node_sizes = (np.log(total_degrees + 2) * 5).clip(min=3).tolist()
    node_text = []

    # Identify sets for faster lookup
    in_set = set(input_idx)
    out_set = set(output_idx)

    for i in range(NUM_NEURONS):
        stats = f"Axons (Out): {int(out_degs[i])}<br>Dendrites (In): {int(in_degs[i])}"

        if i in in_set:
            node_colors[i] = '#00FFFF'  # Inputs: Cyan
            node_sizes[i] = max(node_sizes[i], 12)
            node_text.append(f"<b>INPUT Node {i}</b><br>{stats}")
        elif i in out_set:
            node_colors[i] = '#FF00FF'  # Outputs: Magenta
            node_sizes[i] = max(node_sizes[i], 12)
            node_text.append(f"<b>OUTPUT Node {i}</b><br>{stats}")
        else:
            # Color hidden nodes based on how "Hub-like" they are
            intensity = min(255, int((total_degrees[i] / np.max(total_degrees)) * 200 + 55))
            node_colors[i] = f'rgb({intensity}, {intensity}, {intensity})'
            node_text.append(f"Hidden Node {i}<br>{stats}")

    # Draw Nodes
    trace_nodes = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='black', width=0.5),
            opacity=0.95
        ),
        text=node_text, hoverinfo='text', name='Neurons'
    )

    # Draw Edges
    norm_edge_x, norm_edge_y, norm_edge_z = [], [], []

    for s, d in zip(src, dst):
        norm_edge_x.extend([x[s], x[d], None])
        norm_edge_y.extend([y[s], y[d], None])
        norm_edge_z.extend([z[s], z[d], None])

    trace_norm_edges = go.Scatter3d(
        x=norm_edge_x, y=norm_edge_y, z=norm_edge_z,
        mode='lines', line=dict(color='rgba(150, 150, 150, 0.1)', width=1),
        hoverinfo='none', name='Synapses'
    )

    # Hide grid and axes
    no_axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title='')

    layout = go.Layout(
        title=f'Scale Invariant Neural Graph (SING) - {NUM_NEURONS} Nodes<br>'
              f'<span style="font-size:12px;">Sphere Size = Total Degree | '
              f'<span style="color:#00FFFF;">Cyan=Inputs</span>, <span style="color:#FF00FF;">Magenta=Outputs</span></span>',
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, bgcolor='black'),
        paper_bgcolor='black', font=dict(color='white'), margin=dict(l=0, r=0, b=0, t=60), showlegend=False
    )

    fig = go.Figure(data=[trace_norm_edges, trace_nodes], layout=layout)
    print("Opening browser...")
    fig.show()


if __name__ == "__main__":
    plot_sing()
