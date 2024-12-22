import matplotlib.pyplot as plt


def CrossEntrophyLoss_plot(data1, data2=None, labels=("Data1", "Data2"), title=None, xlabel="Epoch",
                           ylabel="error"):
    """
    Creates a plot with two sets of data on the same graph.

    Parameters:
    - data1 (list of tuples): First dataset, each tuple contains (Epoch (int), error (float)).
    - data2 (list of tuples): Second dataset, each tuple contains (Epoch (int), error (float)).
    - labels (tuple of str): Labels for the two datasets (default: ("Data1", "Data2")).
    - title (str): Title of the graph (default: f"CrossEntrophyLoss - {labels[0]}, {labels[1]}" if data2 is not None else f"EntrophyLoss - {labels[0]}").
    - xlabel (str): Label for the x-axis (default: "Epoch").
    - ylabel (str): Label for the y-axis (default: "CrossEntrophyLoss").
    """
    # Sort data by size for proper plotting
    data1 = sorted(data1, key=lambda x: x[0])
    data2 = sorted(data2, key=lambda x: x[0]) if data2 is not None else None

    # Extract sizes and errors for both datasets
    Epochs1 = [item[0] for item in data1]
    errors1 = [item[1] for item in data1]
    if data2 is not None:
        Epochs2 = [item[0] for item in data2]
        errors2 = [item[1] for item in data2]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot first dataset
    plt.plot(Epochs1, errors1, marker="o", linestyle="-", color="blue", label=labels[0])

    # Plot second dataset
    if data2 is not None:
        plt.plot(Epochs2, errors2, marker="o", linestyle="-", color="red", label=labels[1])

    # Add title, labels, legend, and grid
    if title is None:
        if data2 is None:
            title = f"EntrophyLoss - {labels[0]}"
        else:
            title = f"CrossEntrophyLoss - {labels[0]}, {labels[1]}"
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def main():
    # Example usage
    data1 = [
        (100, 0.45),
        (200, 0.40),
        (500, 0.35),
        (1000, 0.30),
        (1500, 0.28),
        (2000, 0.25),
        (3000, 0.23),
        (4000, 0.22),
        (5000, 0.21),
        (7500, 0.20),
        (10000, 0.19),
        (15000, 0.18),
        (20000, 0.17)
    ]

    data2 = [
        (100, 0.50),
        (200, 0.45),
        (500, 0.40),
        (1000, 0.37),
        (1500, 0.34),
        (2000, 0.30),
        (3000, 0.28),
        (4000, 0.26),
        (5000, 0.24),
        (7500, 0.22),
        (10000, 0.21),
        (15000, 0.20),
        (20000, 0.19)
    ]

    CrossEntrophyLoss_plot(data1, data2, labels=("Train Error", "Validation Error"))
    CrossEntrophyLoss_plot(data1, labels=("Train Error", "Validation Error"))


if __name__ == "__main__":
    main()
