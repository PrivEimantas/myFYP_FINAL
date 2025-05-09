# p2pfl/learning/aggregators/atten.py
#
# Example:
#     from p2pfl.learning.aggregators.atten import Atten
#     agg = Atten(vae_ckpt_path="vae_state_dict.pt",      # your saved VAE
#                 device="cuda",                          # or "cpu"
#                 threshold="mean")                       # "mean", "median", or float
#
# -------------------------------------------------------

from __future__ import annotations
from typing import List, Union

import numpy as np
import torch

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

# -------------------------------------------------------------------- #
#  Bring in the user's VAE class.  Either import it, or copy-paste.    #
#  (Assumes vae.py is on PYTHONPATH; adjust if necessary.)             #
# -------------------------------------------------------------------- #
from test.Experiments.Aggregators.nets import VAE        # ← change to the actual path of your class


class Atten(Aggregator):
    """
    Robust FL aggregator (“atten”) that flags malicious client updates
    with a *pre-trained* PyTorch VAE and averages only the honest ones.
    """

    def __init__(self,
                 vae_ckpt_path: str = "/workspaces/p2pfl/test/Experiments/Aggregators/mnist_vae_16100.pt",
                 device: str = "cpu",
                 threshold: Union[str, float] = "mean"):
        """
        Parameters
        ----------
        vae_ckpt_path : str
            Path to the `.pt` file produced by
            `torch.save(vae.state_dict(), path)`.
        device : str
            "cpu", "cuda", or "cuda:IDX".
        threshold : {"mean","median"} or float
            Rule for converting per-client scores to a boolean mask.
        """
        super().__init__()
        self.partial_aggregation = True          # same semantics as FedAvg
        self.device = torch.device(device)

        # ----- load the trained VAE in eval mode -----
        self.vae = None                     # defaults match your file
        # self.vae.load_state_dict(torch.load(vae_ckpt_path,
        #                                     map_location=self.device))
        # self.vae.to(self.device).eval()

        self.threshold_rule = threshold

    # ---------------- helper : numpy ↔ torch ---------------- #
    @staticmethod
    def _flatten(self,model: P2PFLModel) -> np.ndarray:
        """1-D float32 vector of all layer weights."""
        return np.concatenate([w.ravel() for w in model.get_parameters()]).astype(np.float32)

    @staticmethod
    def _unflatten(self,template: List[np.ndarray], vec: np.ndarray) -> List[np.ndarray]:
        """Split a 1-D vector back to the original layer shapes."""
        out, cursor = [], 0
        for layer in template:
            size = layer.size
            out.append(vec[cursor:cursor + size].reshape(layer.shape))
            cursor += size
        return out

    def _ensure_vae(self, dim):
        if self.vae is None:
            self.vae = VAE(input_dim=dim).to(self.device)
            self.vae.load_state_dict(torch.load("/workspaces/p2pfl/test/Experiments/Aggregators/mnist_vae_16100.pt",
                                                map_location=self.device))
            self.vae.eval()

    # -------------------- main entry-point -------------------- #
    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        flat_first = np.concatenate([w.ravel() for w in models[0].get_parameters()])
        self._ensure_vae(len(flat_first))

        # 1) Stack flattened updates → (K, D)
        flat_updates = [self._flatten(m) for m in models]
        X = np.vstack(flat_updates).astype(np.float32)

        # 2) Compute VAE reconstruction losses (vectorised)
        with torch.no_grad():
            tensor_X   = torch.tensor(X, device=self.device)
            x_hat, z_mu, z_logvar = self.vae.forward(tensor_X)

            # reproduction loss exactly like your test() method
            mse   = torch.nn.functional.mse_loss(x_hat, tensor_X, reduction="none") \
                        .view(tensor_X.size(0), -1).sum(dim=1)
            kld   = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
            scores = (mse + kld).cpu().numpy()           # (K,)

        # 3) Decide threshold
        if isinstance(self.threshold_rule, float):
            thr = self.threshold_rule
        elif self.threshold_rule == "median":
            thr = np.median(scores)
        else:                                           # default: mean
            thr = np.mean(scores)

        malicious = scores > thr                       # boolean vector

        # 4) Sample-count weights, zeroing out bad guys
        weights = np.array([m.get_num_samples() for m in models], dtype=np.float64)
        weights[malicious] = 0.0
        if weights.sum() == 0:
            # fallback: if all flagged, revert to FedAvg weights
            weights = np.array([m.get_num_samples() for m in models], dtype=np.float64)
        weights /= weights.sum()

        # 5) Weighted average in flat space
        weighted_mean = np.average(X, axis=0, weights=weights)

        # 6) Restore layer shapes
        new_params = self._unflatten(models[0].get_parameters(), weighted_mean)

        # 7) Contributors list (only honest clients)
        contributors: List[str] = []
        for is_bad, m in zip(malicious, models):
            if not is_bad:
                contributors.extend(m.get_contributors())

        # 8) Return the new aggregated model
        total_samples = int(np.sum([m.get_num_samples() for m in models]))
        return models[0].build_copy(params=new_params,
                                    num_samples=total_samples,
                                    contributors=contributors)
