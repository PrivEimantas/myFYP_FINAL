import copy
from typing import List, Union

import numpy as np
import torch

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

class BayesianRobustAggregation(Aggregator):
    """
    Bayesian Robust Aggregation (BRA) aggregator.
    
    This aggregator computes a reconstruction loss (using a pre-trained VAE if provided)
    or a fallback error (based on deviation from the average update) for each client update.
    Client updates with score above a threshold are considered malicious and are zeroed
    out during weighted averaging.
    
    Attributes:
        max_em_steps: Maximum number of EM steps (not used in this simple implementation).
        tol: Tolerance for convergence (not used here).
        threshold_rule: "mean", "median", or a float value to decide malicious updates.
        device: Torch device string.
        vae: Pre-trained VAE model (if provided) for scoring client updates.
    """
    def __init__(self,
                 max_em_steps: int = 3,
                 tol: float = 1e-4,
                 threshold: Union[str, float] = "mean",
                 device: str = "cpu") -> None:
        super().__init__()
        self.partial_aggregation = True  # same semantics as FedAvg
        self.max_em_steps = max_em_steps
        self.tol = tol
        self.threshold_rule = threshold
        self.device = device
        self.vae = None  # Optionally, load a pre-trained VAE externally

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # Total samples for weighting
        total_samples = sum(m.get_num_samples() for m in models)
        # Get template weights (assumed to be a list of numpy arrays) from the first model
        template = models[0].get_parameters()

        # Helper: Flatten a list of numpy arrays into a single vector
        def flatten_params(params: List[np.ndarray]) -> np.ndarray:
            return np.concatenate([p.ravel() for p in params])

        # Helper: Unflatten the flat vector back into the list structure of template
        def unflatten_params(template: List[np.ndarray], flat: np.ndarray) -> List[np.ndarray]:
            new_params = []
            start = 0
            for layer in template:
                size = layer.size
                new_params.append(flat[start:start + size].reshape(layer.shape))
                start += size
            return new_params

        # Flatten each model's parameters into a 1-D vector
        flat_models = [flatten_params(m.get_parameters()) for m in models]
        flat_models_arr = np.vstack(flat_models)  # shape: (num_clients, flat_dim)

        # Compute a score for each update using the provided VAE, or a fallback error metric
        if self.vae is not None:
            pass #not needed
            # with torch.no_grad():
            #     tensor_X = torch.tensor(flat_models_arr, dtype=torch.float32, device=self.device)
            #     # Assume the VAE's forward returns: (x_hat, z_mu, z_logvar)
            #     x_hat, z_mu, z_logvar = self.vae(tensor_X)
            #     mse = torch.nn.functional.mse_loss(x_hat, tensor_X, reduction="none") \
            #             .view(tensor_X.size(0), -1).sum(dim=1)
            #     kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar), dim=1)
            #     scores = (mse + kld).cpu().numpy()
        else:
            # Fallback: measure squared deviation from the mean update
            mean_update = np.mean(flat_models_arr, axis=0)
            scores = np.array([np.sum((update - mean_update) ** 2) for update in flat_models_arr])

        # Determine threshold for malicious updates
        if isinstance(self.threshold_rule, float):
            thr = self.threshold_rule
        elif self.threshold_rule == "median":
            thr = np.median(scores)
        else:  # default to mean
            thr = np.mean(scores)

        # Identify malicious updates (scores above threshold)
        malicious = scores > thr

        # Compute sample-weighting, zeroing out malicious updates
        sample_counts = np.array([m.get_num_samples() for m in models], dtype=np.float64)
        sample_counts[malicious] = 0.0
        if sample_counts.sum() == 0:
            sample_counts = np.array([m.get_num_samples() for m in models], dtype=np.float64)
        weights = sample_counts / sample_counts.sum()

        # Compute weighted average in the flat space
        agg_flat = np.zeros(flat_models_arr.shape[1])
        for i, w in enumerate(weights):
            agg_flat += flat_models_arr[i] * w

        # Restore the aggregated parameters to the original shape
        new_params = unflatten_params(template, agg_flat)

        # Gather contributors from non-malicious updates
        contributors: List[str] = []
        for is_bad, m in zip(malicious, models):
            if not is_bad:
                contributors.extend(m.get_contributors())

        return models[0].build_copy(params=new_params, num_samples=total_samples, contributors=contributors)

    def get_required_callbacks(self) -> List[str]:
        return []