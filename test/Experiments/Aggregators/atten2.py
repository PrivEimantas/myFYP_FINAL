#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Federated Averaging (FedAvg) Aggregator."""

from typing import List

import numpy as np
import copy,torch
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from test.Experiments.Aggregators.nets import VAE  # Your VAE model implementation

def to_ndarray(w_locals):
    for user_idx in range(len(w_locals)): 
        for key in w_locals[user_idx]:
            w_locals[user_idx][key] = w_locals[user_idx][key].cpu().numpy()
    return w_locals

class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016].

    Paper: https://arxiv.org/abs/1602.05629.
    """

    def __init__(self) -> None:
        """Initialize the aggregator."""
        super().__init__()
        self.partial_aggregation = True

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")
        
        w_locals = models[0].get_parameters() 
        layer_shape_size = {}
        for key in w_locals[0]:
            layer_shape_size[key] = ( w_locals[0][key].numel(), list(w_locals[0][key].shape) )
        print(layer_shape_size)

        w_locals = to_ndarray(w_locals)


        user_one_d = []
        for user_idx in range(len(w_locals)):
            tmp = np.array([])
            for key in w_locals[user_idx]:
                data_idx_key = np.array(w_locals[user_idx][key]).flatten()
                tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )
            user_one_d.append(tmp)

        sample_users = 20
        # indices of malicious clients (0, 1, 2, 3, 4, 5)
        attacker_idx = list(range(6))
        user_one_d_test = copy.deepcopy(user_one_d)

        model = VAE( input_dim = user_one_d_test[0].shape[0] )
        model.load_state_dict( torch.load("/workspaces/p2pfl/test/Experiments/Aggregators/mnist_vae_16100.pt") )
        model.eval()


        scores = model.test(user_one_d_test)
        score_avg = np.mean(scores)
        print("scores", scores)
        print("score_avg", score_avg)
        pred = scores > score_avg


        ew_weights = copy.deepcopy(user_weights)
        for _ in range(len(pred)):
            if pred[_]:
                new_weights[_] = 0.0
        new_weights = new_weights / sum(new_weights)

        user_one_d = np.array(user_one_d)
        selected = np.zeros(user_one_d[0].shape)
        for _ in range(len(new_weights)):
            selected += user_one_d[_] * new_weights[_]

        #     # Total Samples
        # total_samples = sum([m.get_num_samples() for m in models])
        
        # # Create a Zero Model using numpy
        # first_model_weights = models[0].get_parameters() 
        # accum = [np.zeros_like(layer) for layer in first_model_weights]

        # # Add weighted models
        # for m in models:
        #     for i, layer in enumerate(m.get_parameters()):
        #         accum[i] = np.add(accum[i], layer * m.get_num_samples())

        # # Normalize Accum
        # accum = [np.divide(layer, total_samples) for layer in accum]

        # # Get contributors
        # contributors: List[str] = []
        # for m in models:
        #     contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model
        return models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)
