#
# FoolsGold Aggregator
# Paper: Fung, Yoon, Beschastnikh – “FoolsGold: Countering Sybils in
#        Federated Learning”, CCS 2019 (Alg. 1).
#

from typing import List, Dict
import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FoolsGold(Aggregator):
    """
    Sybil-resilient aggregation.

    Every client keeps a *credibility weight* αᵢ ∈ [0,1]
    that the server updates from the cosine similarity of the
    **historical** gradient directions it has seen so far.

    Parameters
    ----------
    lr   : float   — server learning-rate η  (default 1.0, as in the paper)
    eps  : float   — numerical stabiliser for norms / divisions
    """

    def __init__(self, lr: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.partial_aggregation = True
        self.lr = lr
        self.eps = eps

        # persistent state
        self._history: Dict[str, np.ndarray] = {}      # node_id → ∑ g_t
        self._global_params: List[np.ndarray] | None = None

    # --------------------- helpers ------------------------------------

    def _flatten(self, layers: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([l.ravel() for l in layers])

    def _unflatten(self, vec: np.ndarray,
                   template: List[np.ndarray]) -> List[np.ndarray]:
        out, k = [], 0
        for t in template:
            sz = t.size
            out.append(vec[k:k + sz].reshape(t.shape))
            k += sz
        return out

    # --------------------- main entry ---------------------------------

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:

        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) no models to aggregate")

        # ── 1.  Cold-start global weights (FedAvg) ────────────────────
        if self._global_params is None:
            self._global_params = [np.mean([m.get_parameters()[i] for m in models], axis=0)
                                   for i in range(len(models[0].get_parameters()))]

        # ── 2.  Collect *current-round* gradients gᵢ ──────────────────
        node_ids, round_grads = [], []
        for m in models:
            nid = m.get_contributors()[0]           # assume one owner / model
            node_ids.append(nid)

            try:
                grad_layers = m.get_gradient()
            except AttributeError:                  # fallback → weight delta
                current = m.get_parameters()
                grad_layers = [current[i] - self._global_params[i]
                               for i in range(len(current))]

            g_flat = self._flatten(grad_layers)
            round_grads.append(g_flat)

            # accumulate historical memory  Gᵢ ← Gᵢ + gᵢ
            if nid not in self._history:
                self._history[nid] = np.zeros_like(g_flat)
            self._history[nid] += g_flat

        # ── 3.  Cosine-similarity matrix  C  on historical grads ──────
        H = np.stack([self._history[nid] for nid in node_ids], axis=0)
        H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + self.eps)
        C = H_norm @ H_norm.T
        np.fill_diagonal(C, 0.0)                    # ignore self-similarity

        # ── 4.  FoolsGold weighting  αᵢ = 1 − maxⱼ Cᵢⱼ  (+ “pardoning”) ─
        max_sim = C.max(axis=1)
        alpha = 1.0 - max_sim
        # pardoning step so that at least one honest client keeps α=1
        m = alpha.max()
        if m > 0:
            alpha = alpha / m
        alpha[alpha < 0] = 0.0                      # clip
        alpha = alpha.reshape(-1, 1)                # column for broadcast

        # ── 5.  Aggregate gradient update  w ← w − η · Σ αᵢ gᵢ ─────────
        G = np.stack(round_grads, axis=0)
        agg_grad = (alpha * G).sum(axis=0)
        agg_grad_layers = self._unflatten(agg_grad, self._global_params)

        new_params = [w - self.lr * g
                      for w, g in zip(self._global_params, agg_grad_layers)]
        self._global_params = new_params

        # ── 6.  build P2PFLModel  (metadata) ───────────────────────────
        contributors: List[str] = [cid
                                   for m in models
                                   for cid in m.get_contributors()]
        total_samples = sum(m.get_num_samples() for m in models)

        return models[0].build_copy(params=new_params,
                                    num_samples=total_samples,
                                    contributors=contributors)
