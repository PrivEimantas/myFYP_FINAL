

from typing import List
import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class RFA(Aggregator):
   

    def __init__(self,
                 max_iter: int = 10,
                 tol: float = 1e-5,
                 c: float = 1.345,
                 eps: float = 1e-12) -> None:
        super().__init__()
        self.partial_aggregation = True
        self.max_iter, self.tol, self.c, self.eps = max_iter, tol, c, eps

   

    
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



    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:

        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) no models to aggregate")

       
        X = np.stack(
            [self._flatten(m.get_parameters()) for m in models],
            axis=0
        )                                # shape = (n_clients, d)
        n, d = X.shape

       
        sigma = 1.4826 * np.median(np.abs(X - np.median(X, axis=0, keepdims=True)))
        sigma = max(sigma, self.eps)

      
        w = np.median(X, axis=0)

       
        for _ in range(self.max_iter):
            diff = X - w                                   
            r   = np.linalg.norm(diff, axis=1)              
            # Huber-ψ weights  α_i = min(1, c·σ / r_i) / max(r_i,eps)
            alpha = np.minimum(1.0, (self.c * sigma) / (r + self.eps)) / (r + self.eps)
            alpha /= alpha.sum() + self.eps              

            w_new = (alpha[:, None] * X).sum(axis=0)      
            if np.linalg.norm(w_new - w) <= self.tol:
                w = w_new
                break
            w = w_new

  
        new_layers = self._unflatten(w, models[0].get_parameters())

   
        contributors: List[str] = [cid
                                   for m in models
                                   for cid in m.get_contributors()]
        total_samples = sum(m.get_num_samples() for m in models)

        return models[0].build_copy(params=new_layers,
                                    num_samples=total_samples,
                                    contributors=contributors)
