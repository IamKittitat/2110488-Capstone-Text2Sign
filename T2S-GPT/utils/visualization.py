import matplotlib.pyplot as plt

def plot_loss(loss_file, metrics, title, save_path, y_lim = 10):
    """
    Plots training loss components from a loss file.

    Parameters:
        loss_file (str): Path to the CSV file containing loss values.
        metrics (list of str): List of metric names to be plotted. 
                               These should match the column order in the loss file.

    Example:
        metrics = ['L_X_re', 'L_vq', 'L_budget', 'L_total']
        plot_loss('loss_file.csv', metrics)
    """
    
    metric_data = {metric: [] for metric in metrics}
    
    with open(loss_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            if len(values) != len(metrics):
                print(f"Warning: Line with unexpected number of values: {line.strip()}")
                continue
            
            for i, metric in enumerate(metrics):
                metric_data[metric].append(values[i])
    
    iterations = list(range(1, len(metric_data[metrics[0]]) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.ylim(0, y_lim)
    for metric, values in metric_data.items():
        plt.plot(iterations, values, label=metric)
    
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.savefig(save_path)
    plt.close()
    