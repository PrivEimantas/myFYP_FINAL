#
# Robust Average Gradient Algorithm (RAGA) Aggregator
# Paper: https://arxiv.org/abs/2403.13374  (Alg. 1, Eq. 15)  
#

from typing import List
import numpy as np
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class RAGA(Aggregator):
    """
    RAGA =        multi-step local updates on clients  +
                  geometric-median aggregation on *averaged* gradients.

    Server-side options
    -------------------
    lr          : global learning-rate η_t  (Alg 1 line 13)
    tol         : ‖Δ‖₂ tolerance for Weiszfeld convergence
    max_iter    : hard cap on Weiszfeld iterations
    eps         : tiny constant to avoid /0 
    """

    def __init__(self,
                 lr: float = 1e-2,
                 tol: float = 1e-5,
                 max_iter: int = 50,
                 eps: float = 1e-12):
        super().__init__()
        self.partial_aggregation = True
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps

        # running copy of the global weights w_t
        self._global_params: List[np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    #                        — utilities —
    # ------------------------------------------------------------------ #

   
    def _flatten(self,layers: List[np.ndarray]) -> np.ndarray:
        """List of np arrays → one 1-D vector."""
        return np.concatenate([p.ravel() for p in layers])

    
    def _unflatten(self,vec: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
        """Inverse of _flatten – split vec back into template shapes."""
        out, idx = [], 0
        for t in template:
            sz = t.size
            out.append(vec[idx: idx + sz].reshape(t.shape))
            idx += sz
        return out

    def _geom_median(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Weighted geometric median via Weiszfeld iterations:​:contentReference[oaicite:2]{index=2}

            y_{k+1} = Σ_i (w_i x_i / d_i)  /  Σ_i (w_i / d_i) ,
            d_i = ‖x_i − y_k‖₂  (ε-smoothed).

        Converges for convex objective (sum of ℓ₂-norms).
        """
        y = np.average(X, axis=0, weights=w)            # init = weighted mean
        for _ in range(self.max_iter):
            d = np.linalg.norm(X - y, axis=1) + self.eps
            y_new = (w[:, None] * X / d[:, None]).sum(axis=0) / (w / d).sum()
            if np.linalg.norm(y_new - y) <= self.tol:
                return y_new
            y = y_new
        return y                                        # return last iterate

    # ------------------------------------------------------------------ #
    #                        — main entry —
    # ------------------------------------------------------------------ #

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) no models to aggregate")

        # ---------- 1.  Book-keeping of global weights -----------------
        if self._global_params is None:
            # cold-start with FedAvg mean
            self._global_params = [np.mean([m.get_parameters()[i] for m in models], axis=0)
                                   for i in range(len(models[0].get_parameters()))]

        # ---------- 2.  Collect averaged gradients z_m -----------------
        grads, weights = [], []
        for m in models:
            # The client must attach 1/K Σ_k ∇F_m(w_{k-1})  via  model.set_gradient(...)
            # ... inside the for-loop that builds grads / weights ...
            try:
                g = m.get_gradient()                 # if you later add the helper
            except AttributeError:
                # fallback: finite-difference from weight delta
                current = m.get_parameters()
                g = [(self._global_params[i] - current[i]) / self.lr    # sign matches ∇
                    for i in range(len(current))]
            # grads.append(self._flatten(g))
                       # list[np.ndarray]  (same shapes as params)
            grads.append(self._flatten(g))             # flat 1-D
            weights.append(m.get_num_samples())        # S_m  (Eq 15)
        X = np.stack(grads, axis=0)                    # shape = (M, D)
        w = np.asarray(weights, dtype=np.float64)

        # ---------- 3.  Geometric median (robust aggregate) -----------
        g_med_flat = self._geom_median(X, w)
        g_med = self._unflatten(g_med_flat, self._global_params)

        # ---------- 4.  Global update  w_{t+1} = w_t − η·z_t ----------
        new_params = [w_i - self.lr * g_i for w_i, g_i in zip(self._global_params, g_med)]
        self._global_params = new_params

        # ---------- 5.  Build & return P2PFLModel ---------------------
        contributors = [cid for m in models for cid in m.get_contributors()]
        eff_samples = sum(weights)                     # for logging only

        return models[0].build_copy(params=new_params,
                                    num_samples=eff_samples,
                                    contributors=contributors)
