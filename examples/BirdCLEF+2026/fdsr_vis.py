import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ==========================================
# HYPERPARAMETERS TO PLAY WITH
# ==========================================
NUM_NEURONS = 512  # Keep < 1000 for smooth 3D rendering
NUM_CONNECTIONS = 16  # Connections per neuron

IN_FEATURES = 32
OUT_FEATURES = 32


# ==========================================

def generate_graph(num_neurons, num_connections, in_features, out_features):
    """Executes the exact FDSR initialization logic"""
    torch.manual_seed(42)

    # 1. Spatial Embedding & Topological Flow
    coords = torch.randn(num_neurons, 4)
    # dist = torch.linalg.vector_norm(coords, dim=1, keepdim=True)
    # coords = (coords / (dist + 1e-8)) * (dist + 0)

    distances = torch.linalg.vector_norm(coords, ord=2, dim=1)
    flow_values = 1 / (distances + 100)
    # flow_values = torch.linalg.vector_norm(coords, ord=2, dim=1)
    # flow_values = torch.exp(flow_values * -0.1)

    sorted_indices = torch.argsort(flow_values)
    input_idx = sorted_indices[:in_features]
    output_idx = sorted_indices[-out_features:]

    # 2. Distance and Topology
    d = torch.cdist(coords, coords, p=2.0)
    r = flow_values.unsqueeze(0) / flow_values.unsqueeze(1)
    D = d / r

    # 3. Masking
    D.fill_diagonal_(float('inf'))
    D[:, input_idx] = float('inf')

    # ALL nodes can now send signals
    valid_src_indices = torch.arange(num_neurons)

    # 4. Routing
    src_list, dst_list = [], []
    k = min(num_connections, num_neurons - in_features - 1)

    for i in valid_src_indices:
        best_dst = torch.topk(D[i], k, largest=False).indices
        src_list.append(torch.full((k,), i.item(), dtype=torch.long))
        dst_list.append(best_dst)

    src_tensor = torch.cat(src_list)
    dst_tensor = torch.cat(dst_list)

    return coords.numpy(), flow_values.numpy(), input_idx.numpy(), output_idx.numpy(), src_tensor.numpy(), dst_tensor.numpy()


def plot_fdsr():
    print(f"Generating FDSR Graph with {NUM_NEURONS} nodes...")
    coords, flow_vals, input_idx, output_idx, src, dst = generate_graph(
        NUM_NEURONS, NUM_CONNECTIONS, IN_FEATURES, OUT_FEATURES
    )

    num_nodes, num_dims = coords.shape

    # Project to 3D if necessary
    if num_dims > 3:
        print(f"Projecting {num_dims}D space to 3D using PCA...")
        pca = PCA(n_components=3)
        plot_coords = pca.fit_transform(coords)
    elif num_dims == 2:
        plot_coords = np.pad(coords, ((0, 0), (0, 1)), mode='constant')
    else:
        plot_coords = coords

    x, y, z = plot_coords[:, 0], plot_coords[:, 1], plot_coords[:, 2]

    # Assign node colors
    node_colors = ['#444444'] * NUM_NEURONS
    node_sizes = [4] * NUM_NEURONS
    node_text = []

    log_flow = np.log(flow_vals)
    norm_flow = (log_flow - log_flow.min()) / (log_flow.max() - log_flow.min() + 1e-9)

    is_output_node = np.zeros(NUM_NEURONS, dtype=bool)
    is_output_node[output_idx] = True

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
            r_col = int(norm_flow[i] * 255)
            b_col = int((1 - norm_flow[i]) * 255)
            node_colors[i] = f'rgb({r_col}, 50, {b_col})'
            node_text.append(f"Hidden Node {i}<br>Flow: {flow_vals[i]:.2f}")

    trace_nodes = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, line=dict(color='white', width=0.5), opacity=0.9),
        text=node_text, hoverinfo='text', name='Neurons'
    )

    # Split edges into standard (grey) and lateral-output (magenta)
    norm_edge_x, norm_edge_y, norm_edge_z = [], [], []
    lat_edge_x, lat_edge_y, lat_edge_z = [], [], []

    for s, d in zip(src, dst):
        if is_output_node[s] and is_output_node[d]:
            lat_edge_x.extend([x[s], x[d], None])
            lat_edge_y.extend([y[s], y[d], None])
            lat_edge_z.extend([z[s], z[d], None])
        else:
            norm_edge_x.extend([x[s], x[d], None])
            norm_edge_y.extend([y[s], y[d], None])
            norm_edge_z.extend([z[s], z[d], None])

    trace_norm_edges = go.Scatter3d(
        x=norm_edge_x, y=norm_edge_y, z=norm_edge_z,
        mode='lines', line=dict(color='rgba(150, 150, 150, 0.15)', width=1),
        hoverinfo='none', name='Standard Synapses'
    )

    # Visualize the zero-initialized lateral output edges!
    trace_lat_edges = go.Scatter3d(
        x=lat_edge_x, y=lat_edge_y, z=lat_edge_z,
        mode='lines', line=dict(color='rgba(255, 0, 204, 0.6)', width=2),
        hoverinfo='none', name='Lateral Output Synapses (Zero-Init)'
    )

    no_axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title='',
                   showspikes=False, visible=False)

    layout = go.Layout(
        title=f'FDSR Architecture (Density: {NUM_CONNECTIONS})<br>'
              f'<span style="font-size:12px;">Cyan=Inputs, Magenta=Outputs, Neon Pink Lines=Lateral Output Hub</span>',
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, bgcolor='black'),
        paper_bgcolor='black', font=dict(color='white'), margin=dict(l=0, r=0, b=0, t=50), showlegend=False
    )

    fig = go.Figure(data=[trace_norm_edges, trace_lat_edges, trace_nodes], layout=layout)
    print("Opening browser...")
    fig.show()


if __name__ == "__main__":
    plot_fdsr()
