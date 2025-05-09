
import copy
import numpy as np
from typing import List

import torch

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

# Helper: Convert a model's parameters (dict or list) into a flat 1-D numpy vector.
def flatten_model_parameters(model_params) -> np.ndarray:
    flat = np.array([])
    if isinstance(model_params, dict):
        # sort keys to ensure consistent order
        for key in sorted(model_params.keys()):
            flat = np.hstack((flat, np.array(model_params[key]).flatten()))
    else:
        for layer in model_params:
            flat = np.hstack((flat, np.array(layer).flatten()))
    return flat

# Helper: Build metadata from a model's parameters (to later reshape the flat vector).
def build_layer_shape_metadata(model_params) -> dict:
    metadata = {}
    if isinstance(model_params, dict):
        keys = sorted(model_params.keys())
        for idx, key in enumerate(keys):
            tensor = model_params[key]
            metadata[f"layer{idx}"] = (int(np.prod(tensor.shape)), list(tensor.shape))
    else:
        for idx, layer in enumerate(model_params):
            metadata[f"layer{idx}"] = (int(np.prod(layer.shape)), list(layer.shape))
    return metadata

# Helper: Reshape a flat vector back to a list of arrays matching the original model parameters.
def reshape_from_oneD(template, flat_vector: np.ndarray) -> List[np.ndarray]:
    # Build metadata from template (assumed to be the same type as from build_layer_shape_metadata)
    metadata = build_layer_shape_metadata(template)
    layers = []
    cursor = 0
    for i in range(len(metadata)):
        size, shape = metadata[f"layer{i}"]
        layer_flat = flat_vector[cursor: cursor + size]
        layers.append(layer_flat.reshape(shape))
        cursor += size
    return layers

class KrumAggregator(Aggregator):
    """
    Krum Aggregator
    ----------------
    This aggregator selects the update that minimizes the summed squared distances to its closest 
    (n - f - 2) neighbors, where n is the number of client updates and f is the assumed number of 
    Byzantine (malicious) nodes.
    
    Returns:
        The aggregated model corresponds to the selected update converted back to the model's structure.
    """
    def __init__(self, f: int = 1) -> None:
        super().__init__()
        self.f = f  # number of Byzantine nodes assumed
        self.partial_aggregation = True

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")
            
        # Get a template from the first model.
        template = models[0].get_parameters()
        # Build layer shape metadata for later reshaping.
        # (This metadata order must match the order in flattening.)
        metadata = build_layer_shape_metadata(template)
        
        # Flatten each model’s parameters into a 1-D vector.
        updates = []
        for m in models:
            params = m.get_parameters()
            flat = flatten_model_parameters(params)
            updates.append(flat)
        updates = np.vstack(updates)  # shape: (num_clients, flat_dimension)
        num_clients = updates.shape[0]

        # Compute Krum scores.
        scores = []
        for i in range(num_clients):
            distances = []
            for j in range(num_clients):
                if i != j:
                    # squared Euclidean distance.
                    distances.append(np.linalg.norm(updates[i] - updates[j])**2)
            distances = np.sort(distances)
            # According to the Krum rule, sum the smallest (n - f - 2) distances.
            nb = num_clients - self.f - 2
            if nb < 0:
                nb = 0
            score = np.sum(distances[:nb])
            scores.append(score)
        
        # Select the update with the smallest score.
        best_idx = int(np.argmin(scores))
        selected = updates[best_idx]

        # Reshape the selected flat vector back to the model's parameter structure.
        new_params = reshape_from_oneD(template, selected)

        # Total number of samples (used for weighting) and contributors list.
        total_samples = int(sum(m.get_num_samples() for m in models))
        contributors: List[str] = []
        for m in models:
            contributors.extend(m.get_contributors())
        
        return models[0].build_copy(params=new_params,
                                    num_samples=total_samples,
                                    contributors=contributors)

    def get_required_callbacks(self) -> List[str]:
        return []