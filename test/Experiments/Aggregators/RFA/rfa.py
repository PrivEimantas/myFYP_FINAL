#
# Robust Federated Aggregation (RFA)
# Pillutla, Kakade & Harchaoui – NeurIPS 2022
#
#  • iterative re-weighting à-la Huber M-estimator
#  • breakdown-point: 50 % (like coordinate-median / GeoMed)
#

from typing import List
import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class RFA(Aggregator):
    """
    Robust Federated Aggregation (Algorithm 1 in the paper).

    Parameters
    ----------
    max_iter  : int     — cap on IRLS iterations (default 10)
    tol       : float   — stop when ‖w_{k}-w_{k-1}‖₂ ≤ tol   (default 1e-5)
    c         : float   — Huber tuning const (c≈1.345 for 95 % efficiency)
    eps       : float   — numerical epsilon
    """

    def __init__(self,
                 max_iter: int = 10,
                 tol: float = 1e-5,
                 c: float = 1.345,
                 eps: float = 1e-12) -> None:
        super().__init__()
        self.partial_aggregation = True
        self.max_iter, self.tol, self.c, self.eps = max_iter, tol, c, eps

    # ---------- helper: flatten / unflatten ---------------------------

    
    def _flatten(self,layers: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([l.ravel() for l in layers])

    
    def _unflatten(self,vec: np.ndarray,
                   template: List[np.ndarray]) -> List[np.ndarray]:
        out, k = [], 0
        for t in template:
            sz = t.size
            out.append(vec[k:k+sz].reshape(t.shape))
            k += sz
        return out

    # ---------- main entry --------------------------------------------

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:

        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) no models to aggregate")

        # ---- 0.  collect *vectors* x_i  ---------------------------------
        X = np.stack(
            [self._flatten(m.get_parameters()) for m in models],
            axis=0
        )                                # shape = (n_clients, d)
        n, d = X.shape

        # ---- 1.  robust scale σ via median-of-means ---------------------
        #  σ̂ = 1.4826 · median_{dims} MAD
        sigma = 1.4826 * np.median(np.abs(X - np.median(X, axis=0, keepdims=True)))
        sigma = max(sigma, self.eps)

        # ---- 2.  initialise centre w = coordinate-wise median ----------
        w = np.median(X, axis=0)

        # ---- 3.  IRLS iterations (Alg 1) -------------------------------
        for _ in range(self.max_iter):
            diff = X - w                                    # (n,d)
            r   = np.linalg.norm(diff, axis=1)              # (n,)
            # Huber-ψ weights  α_i = min(1, c·σ / r_i) / max(r_i,eps)
            alpha = np.minimum(1.0, (self.c * sigma) / (r + self.eps)) / (r + self.eps)
            alpha /= alpha.sum() + self.eps                # normalise

            w_new = (alpha[:, None] * X).sum(axis=0)        # weighted mean
            if np.linalg.norm(w_new - w) <= self.tol:
                w = w_new
                break
            w = w_new

        # ---- 4.  reshape back into layer tensors -----------------------
        new_layers = self._unflatten(w, models[0].get_parameters())

        # ---- 5.  metadata / return ------------------------------------
        contributors: List[str] = [cid
                                   for m in models
                                   for cid in m.get_contributors()]
        total_samples = sum(m.get_num_samples() for m in models)

        return models[0].build_copy(params=new_layers,
                                    num_samples=total_samples,
                                    contributors=contributors)
