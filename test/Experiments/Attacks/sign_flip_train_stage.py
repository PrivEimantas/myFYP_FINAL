from test.Experiments.node_registry import get_nodeMap

def apply_sign_flip_to_trainset(state, fraction=0.3) -> None:
    """
    Apply a sign-flip attack on a fraction of nodes in the training set.
    Since state.train_set is a list of node addresses, we use a global registry to look up each node.
    
    Args:
        state: The NodeState.
        fraction: Fraction of nodes to attack.
    """
    
    if state.train_set and state.addr == state.train_set[0]:
        nodes_map = get_nodeMap()
        
        num_attack_nodes = int(len(nodes_map) * fraction)
        print("Sign flip 1")
        for i in range(num_attack_nodes):
            
            
            node = nodes_map[i]
            
            print("Sign flip 3")
            underlying_model = node.get_model().model  
            print("Sign flip 4")
            if hasattr(underlying_model, "get_weights"):
                weights = underlying_model.get_weights()
                print("Starting weight flipping...")
                flipped_weights = [-w for w in weights]
                print("Weight flipping done.")
                underlying_model.set_weights(flipped_weights)
            else:
                weights = underlying_model.state_dict()
                flipped_weights = {k: -v for k, v in weights.items()}
                underlying_model.load_state_dict(flipped_weights)
            print(f"Applied sign flip attack to node {i} in train_set.")