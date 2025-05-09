# Global registry for nodes
nodes_map = []
flipped_flag = False

def register_node(node):
    """Register a node in the global registry."""
    nodes_map.append(node)

def get_nodeMap():
    """Retrieve a node by its address."""
    return nodes_map

def set_flag_true() -> None:
    """Set the flipped flag."""
    global flipped_flag
    flipped_flag = True

def release_flag() -> None:
    """Release the flipped flag."""
    global flipped_flag
    flipped_flag = False

def get_flag() -> bool:
    """Get the flipped flag."""
    return flipped_flag
