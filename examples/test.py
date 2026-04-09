import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class SNN(tt.Model):
    def __init__(self, c, n_labels):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(c, 16, 3),
            snn.LIB(16, 0.9, 1.0, dim=-3),
            nn.Flatten(),
            nn.LazyLinear(16),
            snn.RLIB(16, 0.9, 0.9, 1.0),
            nn.Linear(16, 16),
            snn.SLIB(16, 0.1, 0.9, 1.0),
            nn.Linear(16, 16),
            snn.DLIB(16, 0.1, 0.9, 0.1),
            nn.Linear(16, n_labels),
            snn.LI(n_labels, 0.9),
            nn.Softmax(-1)
        )

    def forward(self, x):
        return self.mlp(x)


b, c, h, w = 10, 3, 28, 28

model = SNN(c, b).to(device)

random_feature = torch.rand(b, c, h, w).to(device)
random_label = torch.eye(b).to(device)

print(f"Feature: {random_feature.shape} | Label: {random_label.shape}")

num_epochs = 100
num_timesteps = 100

optimizer = torch.optim.AdamW(model.parameters(), 1e-2)

losses = []

# loss_fn = snn.functional.mse_loss
loss_fn = tt.loss.soft_cross_entropy  # the more flexible variant of cross-entropy for not only onehot vectors

for epoch in tqdm(range(num_epochs), desc="TRAIN", total=num_epochs):
    model.zero_grad()
    model.train()
    for step in range(num_timesteps):
        model_output = model(random_feature)
    loss = loss_fn(model_output, random_label)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()
    model.zero_states()  # despite the name, actually sets them to None

plt.title("Train loss over time")
plt.plot(losses, label="Train Loss")
plt.legend()
plt.show()

# Let's also take a look at the spike train output that the model makes for each example:
spike_train = []

with torch.no_grad():
    model.eval()
    model.zero_states()
    for step in tqdm(range(num_timesteps), desc="TEST", total=num_timesteps):
        model_output = model(random_feature)
        spike_train.append(model_output)

    for index in range(b):
        curated_spike_train = [tensor[index] for tensor in spike_train]
        curated_loss_train = [loss_fn(tensor[index], random_label[index]) for tensor in spike_train]

        print(f"Example {index} - Expected output: {random_label[index]}")
        tt.plot.line_graph(curated_spike_train, title=f"Example {index} - Output over time")
# tt.plot.line_graph(curated_loss_train, title=f"Example {index} - Loss over time")


# ============================================================================
# COMPILE/decompile TESTING
# ============================================================================

print("\n" + "=" * 80)
print("COMPILE/decompile FUNCTIONALITY TESTING")
print("=" * 80)


def get_model_output(model, input_data, num_timesteps=100):
    """Get model output for consistency checking"""
    model.eval()
    model.zero_states()
    outputs = []
    with torch.no_grad():
        for step in range(num_timesteps):
            output = model(input_data)
            outputs.append(output.detach().clone())
    return outputs


def compare_outputs(outputs1, outputs2, tolerance=1e-5):
    """Compare two sets of outputs within tolerance"""
    if len(outputs1) != len(outputs2):
        return False, f"Length mismatch: {len(outputs1)} vs {len(outputs2)}"

    for i, (out1, out2) in enumerate(zip(outputs1, outputs2)):
        if not torch.allclose(out1, out2, atol=tolerance):
            return False, f"Mismatch at timestep {i}: max diff {torch.max(torch.abs(out1 - out2))}"
    return True, "Outputs match"


# Test 1: Basic compilation and inference
print("\n1. Testing basic compilation and inference...")
print("-" * 50)

# Get baseline output before compilation
baseline_outputs = get_model_output(model, random_feature, num_timesteps=50)

# Compile the model
print("Compiling model...")
model.TTcompile()
print("Model compiled successfully!")

# Check that compilation metadata exists
compiled_layers = []

for name, module in model.named_modules():
    if hasattr(module, '_compiled') and module._compiled:
        compiled_layers.append(type(module).__name__)
print(f"Compiled layers: {len(compiled_layers)} layers compiled")

# Get compiled output
compiled_outputs = get_model_output(model, random_feature, num_timesteps=50)

# Compare outputs
match, msg = compare_outputs(baseline_outputs, compiled_outputs)
print(f"Output consistency after compilation: {'PASS' if match else 'FAIL'} - {msg}")

# Test 2: Save and load compiled model
print("\n2. Testing save/load of compiled model...")
print("-" * 50)

# Save compiled model
compiled_state_dict = model.state_dict()
torch.save(compiled_state_dict, "test_compiled_model.pt")
print("Compiled model saved")

# Create new model and load compiled state
new_model = SNN(c, b).to(device)
new_model.TTcompile()  # Compile first to prepare for loading
new_model.load_state_dict(compiled_state_dict)
print("Compiled model loaded into new instance")

# Test loaded model
loaded_outputs = get_model_output(new_model, random_feature, num_timesteps=50)
match, msg = compare_outputs(compiled_outputs, loaded_outputs)
print(f"Loaded compiled model consistency: {'PASS' if match else 'FAIL'} - {msg}")

# Test 3: Uncompilation and continued training
print("\n3. Testing uncompilation and continued training...")
print("-" * 50)

# decompile the model
print("Uncompiling model...")
model.TTdecompile()
print("Model decompiled successfully!")

# Check that raw parameters are restored
decompiled_layers = []

for name, module in model.named_modules():
    if not hasattr(module, '_compiled'):
        decompiled_layers.append(type(module).__name__)
print(f"decompiled layers: {len(decompiled_layers)} layers restored")

# Get decompiled output
decompiled_outputs = get_model_output(model, random_feature, num_timesteps=50)
match, msg = compare_outputs(baseline_outputs, decompiled_outputs)
print(f"Output consistency after uncompilation: {'PASS' if match else 'FAIL'} - {msg}")

# Test continued training after uncompilation
print("Testing continued training after uncompilation...")
optimizer = torch.optim.AdamW(model.parameters(), 1e-2)
initial_loss = None
for epoch in range(5):  # Quick training test
    model.zero_grad()
    model.train()
    for step in range(num_timesteps):
        model_output = model(random_feature)
    loss = loss_fn(model_output, random_label)
    if initial_loss is None:
        initial_loss = loss.item()
    loss.backward()
    optimizer.step()
    model.zero_states()

print(f"Training after uncompilation: Initial loss {initial_loss:.4f}, Final loss {loss.item():.4f}")
print(f"Training improvement: {'PASS' if loss.item() < initial_loss else 'FAIL'}")

# Test 4: Hidden states save/load functionality
print("\n4. Testing hidden states save/load functionality...")
print("-" * 50)

# Generate some states
model.eval()
model.zero_states()
for step in range(10):  # Generate some states
    _ = model(random_feature)

# Save states
states_before = model.save_states()
print(f"Saved {len(states_before)} hidden states")

# Create new model and load states
model2 = SNN(c, b).to(device)
model2.load_states(states_before, strict=False, device=device)
print("Hidden states loaded successfully")

# Test that states work correctly
model2.eval()
output_with_loaded_states = model2(random_feature)
print("Model with loaded states executed successfully")

# Test 5: Full workflow test
print("\n5. Testing complete workflow: Train -> Compile -> Save -> Load -> Inference -> decompile -> Train")
print("-" * 70)

# Start fresh
workflow_model = SNN(c, b).to(device)

# Train briefly
print("Training model...")
optimizer = torch.optim.AdamW(workflow_model.parameters(), 1e-2)
for epoch in range(10):
    workflow_model.zero_grad()
    workflow_model.train()
    for step in range(num_timesteps):
        workflow_output = workflow_model(random_feature)
    loss = loss_fn(workflow_output, random_label)
    loss.backward()
    optimizer.step()
    workflow_model.zero_states()
training_loss = loss.item()
print(f"Training completed with loss: {training_loss:.4f}")

# Compile and save
print("Compiling and saving...")
workflow_model.TTcompile()
workflow_state = workflow_model.state_dict()
torch.save(workflow_state, "workflow_test_model.pt")

# Load and inference
print("Loading and running inference...")
loaded_workflow_model = SNN(c, b).to(device)
loaded_workflow_model.TTcompile()
loaded_workflow_model.load_state_dict(workflow_state)

inference_outputs = get_model_output(loaded_workflow_model, random_feature, num_timesteps=20)
print(f"Inference completed: {len(inference_outputs)} timesteps generated")

# decompile and continue training
print("Uncompiling and continuing training...")
loaded_workflow_model.TTdecompile()
optimizer = torch.optim.AdamW(loaded_workflow_model.parameters(), 1e-2)

for epoch in range(5):
    loaded_workflow_model.zero_grad()
    loaded_workflow_model.train()
    for step in range(num_timesteps):
        continued_output = loaded_workflow_model(random_feature)
    continued_loss = loss_fn(continued_output, random_label)
    continued_loss.backward()
    optimizer.step()
    loaded_workflow_model.zero_states()

print(f"Continued training loss: {continued_loss.item():.4f}")
print(f"Full workflow test: {'PASS' if continued_loss.item() < training_loss * 1.1 else 'FAIL'}")

# Test 6: Performance comparison
print("\n6. Testing performance comparison between compiled and decompiled...")
print("-" * 60)

import time

# Time decompiled inference
model.eval()
model.zero_states()
start_time = time.time()
for step in range(num_timesteps):
    _ = model(random_feature)
decompiled_time = time.time() - start_time

# Compile and time compiled inference
model.TTcompile()
model.eval()
model.zero_states()
start_time = time.time()
for step in range(num_timesteps):
    _ = model(random_feature)
compiled_time = time.time() - start_time

print(f"decompiled inference time: {decompiled_time:.4f}s")
print(f"Compiled inference time: {compiled_time:.4f}s")
speedup = decompiled_time / compiled_time if compiled_time > 0 else float('inf')
print(f"Speedup factor: {speedup:.2f}x")

# Cleanup
import os

if os.path.exists("test_compiled_model.pt"):
    os.remove("test_compiled_model.pt")
if os.path.exists("workflow_test_model.pt"):
    os.remove("workflow_test_model.pt")

print("\n" + "=" * 80)
print("ALL COMPILE/decompile TESTS COMPLETED")
print("=" * 80)
