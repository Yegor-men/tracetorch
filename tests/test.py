import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

import pytest


class TestLayerEquivalence:
    """Test equivalence between base and flex layer implementations"""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.num_neurons = 10
        self.batch_size = 4
        self.num_timesteps = 100
        self.dim = -1
        self.lr = 1e-2

        self.pos_alpha = torch.rand(self.num_neurons)
        self.neg_alpha = torch.rand(self.num_neurons)
        self.pos_beta = torch.rand(self.num_neurons)
        self.neg_beta = torch.rand(self.num_neurons)
        self.pos_gamma = torch.rand(self.num_neurons)
        self.neg_gamma = torch.rand(self.num_neurons)
        self.pos_threshold = torch.rand(self.num_neurons)
        self.neg_threshold = torch.rand(self.num_neurons)
        self.pos_scale = torch.randn(self.num_neurons)
        self.neg_scale = torch.randn(self.num_neurons)
        self.pos_rec_weight = torch.randn(self.num_neurons)
        self.neg_rec_weight = torch.randn(self.num_neurons)
        self.bias = torch.randn(self.num_neurons) / 3.0

    def create_test_data(self):
        """Create consistent test data for all tests"""
        return torch.randn(self.timesteps, self.batch_size, self.num_neurons)

    def test_li_equivalence(self):
        """Test LI layer equivalence"""
        self._test_layer_equivalence(
            lambda: snn.LI(
                self.num_neurons,
                beta=self.pos_beta,
            ),
            lambda: snn.flex.LI(
                self.num_neurons,
                beta=self.pos_beta,
            ),
            "LI"
        )

    def test_dli_equivalence(self):
        """Test DLI layer equivalence"""
        self._test_layer_equivalence(
            lambda: snn.DLI(
                self.num_neurons,
                pos_beta=self.pos_beta,
                neg_beta=self.neg_beta,
            ),
            lambda: snn.flex.DLI(
                self.num_neurons,
                pos_beta=self.pos_beta,
                neg_beta=self.neg_beta,
            ),
            "DLI"
        )

    def test_sli_equivalence(self):
        """Test SLI layer equivalence"""
        self._test_layer_equivalence(
            lambda: snn.SLI(
                self.num_neurons,
                alpha=self.pos_alpha,
                beta=self.pos_beta,
            ),
            lambda: snn.flex.SLI(
                self.num_neurons,
                alpha=self.pos_alpha,
                beta=self.pos_beta,
            ),
            "SLI"
        )

    def test_rli_equivalence(self):
        """Test RLI layer equivalence"""
        self._test_layer_equivalence(
            lambda: snn.RLI(
                self.num_neurons,
                beta=self.pos_beta,
                gamma=self.pos_gamma,
                rec_weight=self.pos_rec_weight,
                bias=self.bias,
            ),
            lambda: snn.flex.RLI(
                self.num_neurons,
                beta=self.pos_beta,
                gamma=self.pos_gamma,
                rec_weight=self.pos_rec_weight,
                bias=self.bias,
            ),
            "RLI"
        )

    def _test_layer_equivalence(self, base_layer_factory, flex_layer_factory, layer_name):
        """Test equivalence with simpler approach"""

        # Create layers
        base_layer = base_layer_factory()
        flex_layer = flex_layer_factory()

        # Create optimizers
        base_opt = torch.optim.Adam(base_layer.parameters(), lr=self.lr)
        flex_opt = torch.optim.Adam(flex_layer.parameters(), lr=self.lr)

        # Test 1: Forward pass equivalence
        base_layer.zero_states()
        flex_layer.zero_states()

        with torch.no_grad():
            for t in range(self.num_timesteps):
                # Create fresh input for each timestep to avoid graph issues
                x = torch.randn(self.batch_size, self.num_neurons)

                base_out = base_layer(x)
                flex_out = flex_layer(x)

                torch.testing.assert_close(
                    base_out, flex_out,
                    rtol=1e-5, atol=1e-6,
                    msg=f"{layer_name} forward pass mismatch at timestep {t}\nBase:{base_out}\nFlex:{flex_out}"
                )

        # Test 2: Training equivalence (simpler approach)
        base_layer.zero_states()
        flex_layer.zero_states()

        for t in range(self.num_timesteps):
            # Fresh input and target
            x = torch.randn(self.batch_size, self.num_neurons)
            target = torch.randn(self.batch_size, self.num_neurons)

            # Forward pass
            base_out = base_layer(x)
            flex_out = flex_layer(x)

            # Check forward equivalence during training
            torch.testing.assert_close(
                base_out, flex_out,
                rtol=1e-5, atol=1e-6,
                msg=f"{layer_name} training forward mismatch at step {t}\nBase:{base_out}\nFlex:{flex_out}"
            )

            # Calculate loss
            base_loss = nn.functional.mse_loss(base_out, target)
            flex_loss = nn.functional.mse_loss(flex_out, target)

            # Backward pass
            base_layer.zero_grad()
            flex_layer.zero_grad()
            base_loss.backward()
            flex_loss.backward()

            base_layer.detach_states()
            flex_layer.detach_states()

            # Update parameters
            base_opt.step()
            flex_opt.step()
