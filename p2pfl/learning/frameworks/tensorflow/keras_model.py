#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Keras model abstraction for P2PFL."""

from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf  # type: ignore

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
import copy 


#####################
#    KerasModel     #
#####################


class ModelNotBuiltError(Exception):
    """Raised when a model is not built."""

    pass


class KerasModel(P2PFLModel):
    """
    P2PFL model abstraction for TensorFlow/Keras.

    Args:
        model: The Keras model to encapsulate.
        params: Optional initial parameters (list of NumPy arrays or bytes).
        num_samples: Optional number of samples used for training.
        contributors: Optional list of contributor nodes.
        additional_info: Optional dictionary for extra information.
        compression: Optional dictionary for compression settings.

    """

    def __init__(
        self,
        model: tf.keras.Model,
        params: Optional[Union[list[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[list[str]] = None,
        additional_info: Optional[dict[str, Any]] = None,
        compression: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        """Initialize the KerasModel."""
        super().__init__(model, params, num_samples, contributors, additional_info, compression)

        # Ensure the model is built
        if len(model.get_weights()) == 0:  # type: ignore
            raise ModelNotBuiltError(
                "Model must be built before creating a P2PFLMODEL! Please be sure that model.get_weights() return a non empty list."
            )

    def get_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        return self.model.get_weights()

    def set_parameters(self, params: Union[list[np.ndarray], bytes]) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        if isinstance(params, bytes):
            params, additional_info = self.decode_parameters(params)
            self.additional_info.update(additional_info)

        # Set weights layer by layer
        try:
            self.model.set_weights(params)
        except ValueError as e:
            raise ModelNotMatchingError("Parameters don't match the model. Please check the model architecture and the parameters.") from e

    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        return Framework.TENSORFLOW.value


    # def build_copy(self, **kwargs: Any) -> P2PFLModel:
    #     # Extract required parameters from kwargs
    #     params: list[np.ndarray] = kwargs.pop("params")
    #     num_samples: int = kwargs.pop("num_samples")
    #     contributors: list[str] = kwargs.pop("contributors")
        
    #     # Create a deep copy of the underlying Keras model.
    #     new_model = copy.deepcopy(self.model)
        
    #     # Set only the network weights (not optimizer state) with the aggregated parameters.
    #     new_model.set_weights(params)
        
    #     # Recompile the new_model to reinitialize the optimizer state.
    #     new_model.compile(
    #         optimizer=self.model.optimizer,
    #         loss=self.model.loss,
    #         metrics=self.model.metrics,
    #     )
        
    #     # Return a new KerasModel instance wrapping new_model.
    #     return KerasModel(new_model, params=params, num_samples=num_samples, contributors=contributors, **kwargs)