import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ==========================================
# HYPERPARAMETERS TO PLAY WITH
# ==========================================
NUM_NEURONS = 400  # Keep < 1000 for smooth 3D rendering
NUM_CONNECTIONS = 5  # Connections per neuron
NUM_DIMS = 3  # Spatial dimensions (if >3, PCA is used for display)
FLOW = 1  # The flow scalar (Try 0.01, 0.5, and 2.0 to see the shape change!)

IN_FEATURES = 10
OUT_FEATURES = 10


# ==========================================

def generate_graph(num_neurons, num_connections, num_dims, flow, in_features, out_features):
    """Executes the exact FDSR initialization logic"""
    torch.manual_seed(42)  # Fixed seed for reproducible visualization comparisons

    # 1. Spatial Embedding & Topological Flow
    coords = torch.randn(num_neurons, num_dims)
    flow_vals = torch.exp(flow * coords.sum(dim=1))

    sorted_indices = torch.argsort(flow_vals)
    input_idx = sorted_indices[:in_features]
    output_idx = sorted_indices[-out_features:]

    # 2. Distance and Topology
    d = torch.cdist(coords, coords, p=2.0)
    r = flow_vals.unsqueeze(0) / flow_vals.unsqueeze(1)
    D = d / r

    # 3. Masking
    D.fill_diagonal_(float('inf'))
    D[:, input_idx] = float('inf')

    is_valid_src = torch.ones(num_neurons, dtype=torch.bool)
    is_valid_src[output_idx] = False
    valid_src_indices = torch.nonzero(is_valid_src).squeeze(-1)

    # 4. Routing
    src_list, dst_list = [], []
    k = min(num_connections, num_neurons - in_features - 1)

    for i in valid_src_indices:
        best_dst = torch.topk(D[i], k, largest=False).indices
        src_list.append(torch.full((k,), i.item(), dtype=torch.long))
        dst_list.append(best_dst)

    src_tensor = torch.cat(src_list)
    dst_tensor = torch.cat(dst_list)

    return coords.numpy(), flow_vals.numpy(), input_idx.numpy(), output_idx.numpy(), src_tensor.numpy(), dst_tensor.numpy()


def plot_fdsr():
    print(f"Generating FDSR Graph with {NUM_NEURONS} nodes and Flow={FLOW}...")
    coords, flow_vals, input_idx, output_idx, src, dst = generate_graph(
        NUM_NEURONS, NUM_CONNECTIONS, NUM_DIMS, FLOW, IN_FEATURES, OUT_FEATURES
    )

    # Project to 3D if necessary
    if NUM_DIMS > 3:
        print(f"Projecting {NUM_DIMS}D space to 3D using PCA...")
        pca = PCA(n_components=3)
        plot_coords = pca.fit_transform(coords)
    elif NUM_DIMS == 2:
        plot_coords = np.pad(coords, ((0, 0), (0, 1)), mode='constant')  # Add Z=0
    else:
        plot_coords = coords

    x, y, z = plot_coords[:, 0], plot_coords[:, 1], plot_coords[:, 2]

    # Assign node colors
    node_colors = ['#444444'] * NUM_NEURONS  # Default hidden: grey
    node_sizes = [4] * NUM_NEURONS
    node_text = []

    # Color mapping based on normalized flow for hidden nodes
    log_flow = np.log(flow_vals)
    norm_flow = (log_flow - log_flow.min()) / (log_flow.max() - log_flow.min() + 1e-9)

    for i in range(NUM_NEURONS):
        if i in input_idx:
            node_colors[i] = '#00FFCC'  # Inputs: Cyan
            node_sizes[i] = 8
            node_text.append(f"<b>INPUT</b> Node {i}<br>Flow: {flow_vals[i]:.2f}")
        elif i in output_idx:
            node_colors[i] = '#FF00CC'  # Outputs: Magenta
            node_sizes[i] = 8
            node_text.append(f"<b>OUTPUT</b> Node {i}<br>Flow: {flow_vals[i]:.2f}")
        else:
            # Color hidden nodes on a blue-to-red scale based on flow
            r_col = int(norm_flow[i] * 255)
            b_col = int((1 - norm_flow[i]) * 255)
            node_colors[i] = f'rgb({r_col}, 50, {b_col})'
            node_text.append(f"Hidden Node {i}<br>Flow: {flow_vals[i]:.2f}")

    # Create the Nodes Trace
    trace_nodes = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='white', width=0.5),
            opacity=0.9
        ),
        text=node_text,
        hoverinfo='text',
        name='Neurons'
    )

    # Create the Edges Trace
    # Plotly trick: To draw thousands of lines efficiently, we use a single trace separated by None
    edge_x, edge_y, edge_z = [], [], []
    for s, d in zip(src, dst):
        edge_x.extend([x[s], x[d], None])
        edge_y.extend([y[s], y[d], None])
        edge_z.extend([z[s], z[d], None])

    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.15)', width=1),
        hoverinfo='none',
        name='Synapses'
    )

    # Render Layout
    layout = go.Layout(
        title=f'FDSR Architecture (Dims: {NUM_DIMS}, Flow: {FLOW}, Density: {NUM_CONNECTIONS})<br>'
              f'<span style="font-size:12px;">Cyan=Inputs, Magenta=Outputs, Color Gradient=Flow Direction</span>',
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='black'  # Dark mode makes the neon colors pop
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False
    )

    fig = go.Figure(data=[trace_edges, trace_nodes], layout=layout)
    print("Opening browser...")
    fig.show()


if __name__ == "__main__":
    plot_fdsr()
