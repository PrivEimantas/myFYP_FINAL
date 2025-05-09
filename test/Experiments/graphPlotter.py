import matplotlib.pyplot as plt

def plot_accuracy(metric_dict, title="Accuracy vs Rounds", save_path=None):
    """
    Plots a graph of accuracy against rounds using a given dictionary.
    
    Args:
        metric_dict (dict): Dictionary with round numbers as keys and accuracy as values.
        title (str, optional): Title of the graph. Defaults to "Accuracy vs Rounds".
        save_path (str, optional): If provided, saves the graph to this path.
        
    Returns:
        None
    """
    # Sort by round (the dictionary keys)
    rounds = sorted(metric_dict.keys())
    accuracies = [metric_dict[r] for r in rounds]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example input: replace with your actual accuracy dictionary
    sample_accuracy = {
        0: 0.05,
        1: 0.525,
        2: 0.575,
        3: 0.65,
        4: 0.70,
        5: 0.725,
        6: 0.70,
        7: 0.65,
        8: 0.70,
        9: 0.72,
        10: 0.90
    }
    plot_accuracy(sample_accuracy, title="Accuracy vs Rounds (Sample Data)")