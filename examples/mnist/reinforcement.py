# ======================================================================================================================

import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ======================================================================================================================

class MNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                lambda x: x * (max_val - min_val) + min_val + offset,
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten to [784]
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)


# ======================================================================================================================

num_epochs = 10
num_timesteps = 1

min_prob, max_prob, noise_offset = 0.0, 1.0, 0.0

train_dataset = MNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = MNIST(train=False, min_val=0.0, max_val=1.0, offset=0.0)

batch_size = 100
num_to_visualize = 5

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# ======================================================================================================================
def foobar(x):
    return nn.functional.sigmoid(2 * x)


class Layer(snn.TTModel):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.lif = snn.LIB(
            out_features,
            beta=torch.rand(out_features),
            threshold=torch.rand(out_features),
            spike_fn=foobar,
            deterministic=False,
            return_probs=True,
        )

        # Critic network - predicts reward for each neuron
        self.critic = nn.Sequential(
            nn.Linear(in_features, out_features // 2),
            nn.ReLU(),
            nn.Linear(out_features // 2, out_features)
        )

        self.logprobs = []
        self.predicted_rewards = []

    def forward(self, x):
        spk, spk_prob, pr_sample = self.lif(self.lin(x))

        # Predict reward for this layer
        predicted_reward = self.critic(x.detach()).mean(dim=-1)  # [B] - average across neurons

        # Store for advantage calculation
        self.predicted_rewards.append(predicted_reward)

        logprob = -torch.log(pr_sample).mean(dim=-1)  # [B]
        self.logprobs.append(logprob)

        return spk


class Decoder(snn.TTModel):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            snn.LI(
                in_features,
                beta=torch.rand(in_features),
            ),
            nn.Linear(in_features, 10),
            nn.Softmax(-1),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

        # Critic for final layer
        self.critic = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 1)  # Single reward prediction
        )

        self.logprobs = []
        self.predicted_rewards = []

    def forward(self, x):
        prob_dist = self.net(x)

        # Predict reward
        predicted_reward = self.critic(x.detach()).squeeze(-1)  # [B]
        self.predicted_rewards.append(predicted_reward)

        sample_indices = torch.multinomial(prob_dist, 1).squeeze(-1)
        onehot_sample = nn.functional.one_hot(sample_indices, num_classes=10).float()

        sampled_probs = prob_dist.gather(1, sample_indices.unsqueeze(-1)).squeeze(-1)
        logprob = -torch.log(sampled_probs + 1e-8)

        self.logprobs.append(logprob)

        return onehot_sample


class SNN(snn.TTModel):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            Layer(784, hidden_dim),
            Layer(hidden_dim, hidden_dim),
            # Layer(hidden_dim, hidden_dim),
            # Layer(hidden_dim, hidden_dim),
            # Layer(hidden_dim, hidden_dim),
            Decoder(hidden_dim, 10),
        )

    def forward(self, x):
        return self.net(x)


model = SNN(512).to(device)
total_params = sum(p.numel() for p in model.parameters())
snn_params = model.get_param_count()
print(f"Total: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

train_losses, train_accs = [], []

for e in range(num_epochs):
    model.train()

    for (image, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        image, label = image.to(device), label.to(device)

        model.zero_grad()
        model.zero_states()

        # Clear logprobs at start of each episode
        for layer in model.net:
            layer.logprobs = []
            layer.predicted_rewards = []

        # Single timestep - see whole image at once
        model_output = model(torch.bernoulli(image))

        # Calculate reward based on classification accuracy
        pred_classes = model_output.argmax(dim=-1)
        true_classes = label.argmax(dim=-1) if label.dim() > 1 else label
        accuracy = (pred_classes == true_classes).float()  # [B]
        actual_reward = torch.where(accuracy == 1.0,
                                    torch.tensor(1.0, device=device),
                                    torch.tensor(-0.1, device=device))  # [B]

        # REINFORCE loss for all layers
        reinforce_loss = 0
        critic_loss = 0
        for layer in model.net:
            for i, (timestep_logprobs, predicted_reward) in enumerate(zip(layer.logprobs, layer.predicted_rewards)):
                # Advantage = actual_reward - predicted_reward
                advantage = actual_reward - predicted_reward

                # Actor loss (REINFORCE with advantage)
                reinforce_loss += (timestep_logprobs * advantage).mean()

                # Critic loss (MSE between predicted and actual reward)
                critic_loss += nn.functional.mse_loss(predicted_reward, actual_reward)

        # Total loss = actor + critic
        total_loss = reinforce_loss + 0.5 * critic_loss

        frac_correct = (pred_classes == true_classes).sum().item() / batch_size

        train_losses.append(total_loss.item())
        train_accs.append(frac_correct)

        total_loss.backward()
        optimizer.step()

    plt.title("LOSS")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    plt.title("ACC")
    plt.plot(train_accs, label="train")
    plt.legend()
    plt.show()

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (image, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            image, label = image.to(device), label.to(device)

            model.zero_states()

            model_output = model(torch.bernoulli(image))

            pred_classes = model_output.argmax(dim=-1)
            true_classes = label.argmax(dim=-1) if label.dim() > 1 else label
            frac_correct = (pred_classes == true_classes).sum().item() / batch_size
            test_acc += frac_correct

        test_acc /= len(test_dataloader)

        print(f"TEST - Acc: {test_acc}")
