import copy
from typing import List, Union

import numpy as np
import torch

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

class BayesianRobustAggregation(Aggregator):
   
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

        
        total_samples = sum(m.get_num_samples() for m in models)
       
        template = models[0].get_parameters()

        
        def flatten_params(params: List[np.ndarray]) -> np.ndarray:
            return np.concatenate([p.ravel() for p in params])

        
        def unflatten_params(template: List[np.ndarray], flat: np.ndarray) -> List[np.ndarray]:
            new_params = []
            start = 0
            for layer in template:
                size = layer.size
                new_params.append(flat[start:start + size].reshape(layer.shape))
                start += size
            return new_params

      
        flat_models = [flatten_params(m.get_parameters()) for m in models]
        flat_models_arr = np.vstack(flat_models)  # shape: (num_clients, flat_dim)

        
        if self.vae is not None:
            pass #not needed
           
        else:
           
            mean_update = np.mean(flat_models_arr, axis=0)
            scores = np.array([np.sum((update - mean_update) ** 2) for update in flat_models_arr])

        
        if isinstance(self.threshold_rule, float):
            thr = self.threshold_rule
        elif self.threshold_rule == "median":
            thr = np.median(scores)
        else:  # default to mean
            thr = np.mean(scores)

        
        malicious = scores > thr

        
        sample_counts = np.array([m.get_num_samples() for m in models], dtype=np.float64)
        sample_counts[malicious] = 0.0
        if sample_counts.sum() == 0:
            sample_counts = np.array([m.get_num_samples() for m in models], dtype=np.float64)
        weights = sample_counts / sample_counts.sum()

       
        agg_flat = np.zeros(flat_models_arr.shape[1])
        for i, w in enumerate(weights):
            agg_flat += flat_models_arr[i] * w

       
        new_params = unflatten_params(template, agg_flat)

        
        contributors: List[str] = []
        for is_bad, m in zip(malicious, models):
            if not is_bad:
                contributors.extend(m.get_contributors())

        return models[0].build_copy(params=new_params, num_samples=total_samples, contributors=contributors)

    def get_required_callbacks(self) -> List[str]:
        return []
