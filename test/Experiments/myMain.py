import contextlib

import numpy as np
# import pytest  # noqa: E402, I001
# from datasets import DatasetDict, load_dataset  # noqa: E402, I001

# from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
# from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol as MemoryCommunicationProtocol
# from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol as MemoryCommunicationProtocol

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset  # noqa: E402
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy, RandomIIDPartitionStrategy
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.management.logger import logger
from p2pfl.node import Node  # noqa: E402

from p2pfl.settings import Settings
from p2pfl.utils.check_ray import ray_installed
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings,wait_convergence,wait_to_finish


# from datasets.utils.logging import disable_progress_bar

# from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch
# from p2pfl.learning.frameworks.lightning_model import LightningModel

# disable_progress_bar()

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch


set_standalone_settings()



import random
import numpy as np
import torch

from test.Experiments.node_registry import register_node



def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic operations in PyTorch:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""
                ATTACKS
                ______________
                |          |
                |          |        



"""
def __train_with_sign_flip(s, n, r, model_build_fn, disable_ray: bool = False, fraction: float=0.3):
    """
    Train the network with a sign-flip attack on one node.
    This function is similar to __train_with_seed but after node creation,
    it flips the sign of the weights for the designated attack node.
    """
    # Configure Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False
    assert ray_installed() != disable_ray

    # Set seed and load data
    Settings.general.SEED = s
    # set_all_seeds(s)
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Create and start nodes
    nodes = []
    # from test.Experiments.Aggregators.atten import Atten
    from test.Experiments.Aggregators.Krum.krum import KrumAggregator
    from test.Experiments.Aggregators.BRA.bra import BayesianRobustAggregation
    from test.Experiments.Aggregators.RAGA.raga import RAGA
    from test.Experiments.Aggregators.FoolsGold.foolsgold import FoolsGold

    from test.Experiments.Aggregators.RFA.rfa import RFA


    numAttackingNodes = int(n * fraction)

    # bra = BayesianRobustAggregation(max_em_steps=3, tol=1e-4)

    for i in range(n):
        mem_protocl = MemoryCommunicationProtocol()
        # fixed_addr = f"127.0.0.1:{6000 + i}"
        node = Node(model_build_fn(), partitions[i],protocol=mem_protocl,aggregator=RFA())
        register_node(node)
        node.start()
        nodes.append(node)

    
    

    # Apply the sign-flip attack deterministically.
    numOfAttackNodes = int(n*fraction)
    for i in range(numOfAttackNodes):
        # Get the model of the attack node
        adversary = nodes[i]
        underlying_model = adversary.get_model().model  
        if hasattr(underlying_model, "get_weights"):
            weights = underlying_model.get_weights()
            # Flip the signs of the weights:
            flipped_weights = [-w for w in weights]
            underlying_model.set_weights(flipped_weights)
        else:
            # Assume it's PyTorch:
            weights = underlying_model.state_dict()
            flipped_weights = [-v for v in weights.values()]
            underlying_model.load_state_dict(flipped_weights)

    # Connect nodes in a star topology
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start global learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)

    # Wait for learning to finish and then stop the nodes.
    wait_to_finish(nodes, timeout=2000)
    [node.stop() for node in nodes]

    return exp_name


def __train_with_additive_noise(s, n, r, model_build_fn, disable_ray: bool = False, attack_node_idx=0, noise_std=0.1):
    """
    Train the network with an additive noise attack on one node.
    Instead of flipping signs, this function adds Gaussian noise (with standard deviation `noise_std`)
    to the designated adversary node's weights.
    """
    # Configure Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False
    assert ray_installed() != disable_ray

    # Set seed and load data
    Settings.general.SEED = s
    # set_all_seeds(s)
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 40, RandomIIDPartitionStrategy)

    # Create and start nodes
    from test.Experiments.Aggregators.Krum.krum import KrumAggregator
    from test.Experiments.Aggregators.BRA.bra import BayesianRobustAggregation
    from test.Experiments.Aggregators.FoolsGold.foolsgold import FoolsGold

    nodes = []
    for i in range(n):
        mem_protocl = MemoryCommunicationProtocol()
        # fixed_addr = f"127.0.0.1:{6000 + i}"
        node = Node(model_build_fn(), partitions[i], protocol=mem_protocl,aggregator=FoolsGold())
        register_node(node)
        node.start()
        nodes.append(node)

    # # Apply the additive noise attack deterministically.
    # if 0 <= attack_node_idx < len(nodes):
    #     adversary = nodes[attack_node_idx]
    #     underlying_model = adversary.get_model().model
    #     weights = underlying_model.state_dict()
    #     import torch
    #     noised_weights = {}
    #     for k, v in weights.items():
    #         noise = torch.randn_like(v) * noise_std
    #         noised_weights[k] = v + noise
    #     underlying_model.load_state_dict(noised_weights)

    # Connect nodes in a star topology
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start global learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)
    wait_to_finish(nodes, timeout=2000)
    [node.stop() for node in nodes]

    return exp_name



def __train_with_adversary(s, n, r, model_build_fn, disable_ray: bool = False, fraction: float = 0.3):
    """
    Train the network where one node acts as an adversary by injecting poisoning during training.
    The adversary node (at index adversary_index) uses the AdversaryNode class.
    Other nodes remain as standard Node.
    """
    # Configure Ray
    if disable_ray:
        from p2pfl.settings import Settings
        Settings.general.DISABLE_RAY = True
    else:
        from p2pfl.settings import Settings
        Settings.general.DISABLE_RAY = False
    from p2pfl.utils.check_ray import ray_installed
    assert ray_installed() != disable_ray

    # Set seed and load data
    # from p2pfl.utils.utils import set_all_seeds
    # from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
    # from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
    Settings.general.SEED = s
    # set_all_seeds(s)
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")

    from test.Experiments.poisonedDataset import poison_partition
    poisonedData = poison_partition(data, poison_fraction=1.0)
    
    numAttackNodes = int(n * fraction)
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # from test.myExperiments.poisonedDataset import poison_partition
    # poisonedPartitions = [poison_partition(part, poison_fraction=1.0) for part in partitions]
    poisonedPartitions = poisonedData.generate_partitions(n * 50, RandomIIDPartitionStrategy)
    # For example, after generating the partitions:
    # original_counts = count_labels_in_partition(partitions[0], partitions[0]._train_split_name)
    # poisoned_counts = count_labels_in_partition(poisonedPartitions[0], poisonedPartitions[0]._train_split_name)
    # print("Original partition label counts:", original_counts)
    # print("Poisoned partition label counts:", poisoned_counts)

    # Create and start nodes, using AdversaryNode for the designated adversary_index.
    from test.Experiments.Aggregators.Krum.krum import KrumAggregator
    from test.Experiments.Aggregators.BRA.bra import BayesianRobustAggregation
    from test.Experiments.Aggregators.FoolsGold.foolsgold import FoolsGold

    nodes = []
    for i in range(n):
        mem_protocl = MemoryCommunicationProtocol()

        if i < numAttackNodes:
            node = Node(model_build_fn(), poisonedPartitions[i], protocol=mem_protocl,aggregator=FoolsGold())
        else:
            node = Node(model_build_fn(), partitions[i],protocol=mem_protocl,aggregator=FoolsGold())
        node.state.node_index=i
        node.start()

        nodes.append(node)

    # Connect nodes in a star topology
    # from p2pfl.utils.topologies import TopologyFactory, TopologyType
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start global learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)

    # Wait for learning to finish and then stop the nodes.
    from p2pfl.utils.utils import wait_to_finish
    wait_to_finish(nodes, timeout=2000)
    [node.stop() for node in nodes]

    return exp_name

"""
    REGULAR TRAINING

"""



        
def __train_with_seed(s, n, r, model_build_fn, disable_ray: bool = False):
    # Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False

    assert ray_installed() != disable_ray

    # Seed
    Settings.general.SEED = s
    # set_all_seeds(s)

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")

    # remove *50 for reduced dataset
    partitions = data.generate_partitions(n*50 , RandomIIDPartitionStrategy)

    
    # Node Creation
    nodes = []
    for i in range(n):
        mem_protocl = MemoryCommunicationProtocol()
        # fixed_addr = f"127.0.0.1:{6000 + i}"
        node = Node(model_build_fn(), partitions[i], protocol=mem_protocl)
        # node.name = f"node_{i}"
        register_node(node)
        node.start()
        nodes.append(node)

    # Connect the nodes
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start Learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)

    # Wait
    wait_to_finish(nodes, timeout=1000)

    # Stop Nodes
    [n.stop() for n in nodes]

    return exp_name



"""
                HELPER FUNCTIONS 
                
                
                
"""
def __get_results(exp_name):
    # Get global metrics
    global_metrics = logger.get_global_logs()[exp_name]
    print(global_metrics)
    # Sort by node name
    global_metrics = dict(sorted(global_metrics.items(), key=lambda item: item[0]))
    # Get only the metrics
    global_metrics = list(global_metrics.values())

    # Get local metrics
    local_metrics = list(logger.get_local_logs()[exp_name].values())
    # Sort by node name and remove it
    local_metrics = [list(dict(sorted(r.items(), key=lambda item: item[0])).values()) for r in local_metrics]

    # Assert if empty
    if len(local_metrics) == 0:
        raise ValueError("No local metrics found")
    if len(global_metrics) == 0:
        raise ValueError("No global metrics found")

    # Return metrics
    return global_metrics, local_metrics

def __flatten_results(item):
    """
    Recursively flatten a nested structure extracting numerical values.
    """
    if isinstance(item, (int, float)):
        return [item]
    elif isinstance(item, list):
        return [sub_item for element in item for sub_item in __flatten_results(element)]
    elif isinstance(item, dict):
        return [sub_item for value in item.values() for sub_item in __flatten_results(value)]
    elif isinstance(item, tuple):
        return [sub_item for element in item for sub_item in __flatten_results(element)]
    else:
        return []

def __get_first_node_results(exp_name):
    # Get global metrics and sort by node name
    global_metrics = logger.get_global_logs()[exp_name]
    global_metrics = dict(sorted(global_metrics.items(), key=lambda item: item[0]))
    # Return values for the first node in the sorted order
    # (Assuming each metric value is a dict containing numerical metrics)
    first_node_metrics = list(global_metrics.values())[0]

    return first_node_metrics




def __get_node_results(exp_name, node_name="node_0"):
    # Get global metrics and filter by our fixed node name
    global_metrics = logger.get_global_logs()[exp_name]
    # Assumes the keys are the fixed node names
    if node_name not in global_metrics:
        raise ValueError(f"No metrics found for node {node_name}")
    return global_metrics[node_name]

def __extract_test_metric(first_node_metrics):
    """
    Extracts and flattens the "test_metric" values from first_node_metrics.
    """
    # test_metric = first_node_metrics.get("test_metric", [])
    test_metric = first_node_metrics.get("compile_metrics", [])
    return __flatten_results(test_metric)

def get_aggregated_test_metric(exp_name, expected_rounds, divisor=3):
    """
    Returns a list of tuples (round_index, aggregated_value) where for each round,
    the function collects the compile_metric value (the second element of each tuple)
    from each node and computes:
      aggregated_value = (sum of all node values for that round) / divisor

    For example, if allNodeMetrics is:
        [
          [(0, 0.08), (1, 0.61), (2, 0.66)],
          [(0, 0.14), (1, 0.77), (2, 0.68)]
        ]
    Then for round 0, it computes aggregated_value = (0.08 + 0.14) / 3
    """
    global_metrics = logger.get_global_logs()[exp_name]
    all_nodes_metrics = []
    rounds=0
    for addr, metrics in global_metrics.items():
        rounds+=1
        compile_metrics = metrics.get("compile_metrics", [])
        if compile_metrics:
            all_nodes_metrics.append(compile_metrics)
    
    if not all_nodes_metrics:
        return None

    aggregated_rounds = []
    for round_index in range(rounds):
        round_values = []
        for node_metrics in all_nodes_metrics:
            if len(node_metrics) > round_index:
                round_values.append(node_metrics[round_index][1])
        if round_values:
            aggregated_value = sum(round_values) / divisor
            aggregated_rounds.append((round_index, aggregated_value))
        else:
            aggregated_rounds.append((round_index, None))
    return aggregated_rounds



def __train_with_seed_2(s, n, r, model_build_fn, disable_ray: bool = False):
    # Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False

    assert ray_installed() != disable_ray

    # Seed
    Settings.general.SEED = s
    set_all_seeds(s)

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(model_build_fn(), partitions[i])
        node.start()
        nodes.append(node)

    # Connect the nodes
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start Learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)

    # Wait
    wait_to_finish(nodes, timeout=1000)

    # Stop Nodes
    [n.stop() for n in nodes]

    return exp_name

def NoAttack():


    n, r = 20, 50

    # model_build_fn=model_build_fn_pytorch
    model_build_fn = model_build_fn_tensorflow


    #train_stage has been modified
    
    # exp_name1 = __train_with_seed(666, n, r, model_build_fn, True) #default
    exp_name1 = __train_with_sign_flip(666, n, r, model_build_fn, True,fraction=0.3) #default
    # exp_name1 = __train_with_additive_noise(666, n, r, model_build_fn, True, attack_node_idx=0, noise_std=0.1) #default
    # exp_name1 = __train_with_adversary(666, n, r, model_build_fn, True,fraction=0.3) 



    global_metrics = logger.get_global_logs()[exp_name1]

    import time
    time.sleep(5) 

    global_metrics = logger.get_global_logs()[exp_name1]
    global_metrics = dict(sorted(global_metrics.items(), key=lambda item: item[0]))

    aggregatedSums = {}
    aggregatedCounts = {}

    for node_addr, metrics in global_metrics.items():
        compile_metrics = metrics.get("compile_metrics", [])
        print("Compile metrics:", compile_metrics)
        for round_values in compile_metrics:
            round_index, value = round_values
            aggregatedSums[round_index] = aggregatedSums.get(round_index, 0) + value
            aggregatedCounts[round_index] = aggregatedCounts.get(round_index, 0) + 1

        aggregatedAverages = {
            round_index: aggregatedSums[round_index] / aggregatedCounts[round_index]
            for round_index in aggregatedSums
        }

    print("Average compile metric per round:", aggregatedAverages)
    # global_metrics = logger.get_global_logs()[exp_name1]
    # global_metrics = dict(sorted(global_metrics.items(), key=lambda item: item[0]))
    # aggregatedValues = dict()
    # count=0
    # for node_addr, metrics in global_metrics.items():
    #     count+=1
    #     compile_metrics = metrics.get("compile_metrics", [])
    #     print("Compile metrics:", compile_metrics)
    #     for round_values in compile_metrics:
    #         round_index = round_values[0]
    #         value = round_values[1]
    #         if round_values[0] not in aggregatedValues:
    #             aggregatedValues[round_index] = value
    #         else:
    #             aggregatedValues[round_index] += value
    # print("Summed up,:",aggregatedValues)

                
    # for key in aggregatedValues:
    #     aggregatedValues[key] = aggregatedValues[key] / n

    # print("Aggregated average:",aggregatedValues)
    # for key, value in aggregatedValues.items():
    #     aggregatedValues[key] = aggregatedValues[key] / aggregatedValues[value]

    # output = []
    # for i in range(r+1):
    #     avg = np.mean([x[1] for x in allNodeMetrics])
    #     output.append(avg)
    #     # print(f"Node {node_addr} compile_metrics: {compile_metrics}")



    # import time
    # time.sleep(1)
    # aggregated_test_metrics = get_aggregated_test_metric(exp_name1,expected_rounds=r)
    # if aggregated_test_metrics:
    #     for round_index, val in enumerate(aggregated_test_metrics):
    #         print(f"Aggregated test metric for round {round_index}: {val}")
    # else:
    #     print("No aggregated test metrics found.")

    # node_addr_list = sorted(global_metrics.keys())
    # print("Node addresses:", node_addr_list)
    # node_addr = list(global_metrics.keys())[0]
    # first_node_addr = node_addr_list[0]
    # first_node_metrics = global_metrics[first_node_addr]
    
    # first_node_raw_1 = __get_first_node_results(exp_name1)
    # test_metric_1 = __extract_test_metric(node_metrics)

    # aggregated_test_metric = get_aggregated_test_metric(exp_name1)
    # print("Final aggregated test metric:", aggregated_test_metric)
    #Prints a list of test_metric values per round
    # print("First experiment test_metric values:", test_metric_1)

    # first_node_raw_2 = __get_first_node_results(exp_name2)
    # test_metric_2 = __extract_test_metric(first_node_raw_2)
    # print("Second experiment test_metric values:", test_metric_2)


if __name__ == "__main__":
    NoAttack()