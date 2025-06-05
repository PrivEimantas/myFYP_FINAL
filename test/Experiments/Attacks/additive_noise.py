from test.Experiments.node_registry import get_nodeMap
import torch
import numpy as np

def apply_additive_additive_noise(state, fraction=0.3) -> None:
    if state.train_set and state.addr == state.train_set[0]:
        nodes_map = get_nodeMap()
        noise_std = 0.5  # Standard deviation of the noise
        num_attack_nodes = int(len(nodes_map) * fraction)
        
        for i in range(num_attack_nodes):
            node = nodes_map[i]
            underlying_model = node.get_model().model  
            
            if hasattr(underlying_model, "get_weights"):
                print("Applying additive noise attack...")
                weights = underlying_model.get_weights()
                noised_weights = []
                for weight in weights:
                  
                    weight_tensor = torch.tensor(weight)
                    noise = torch.randn_like(weight_tensor) * noise_std
                 
                    noised_weights.append((weight_tensor + noise).numpy())
                underlying_model.set_weights(noised_weights)
            
            else:
                weights = underlying_model.state_dict()
                noised_weights = {k: v + torch.randn_like(v) * noise_std for k, v in weights.items()}
                underlying_model.load_state_dict(noised_weights)
            
            print(f"Applied additive noise attack to node {i} in train_set.")
