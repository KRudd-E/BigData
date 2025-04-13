import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix(cm, class_labels, save_path=None, show=True):
    """Make confusion matrix plot that can save/show/both.
    """
    # Make figure first - sets up drawing area
    plt.figure(figsize=(10, 8))  
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.0f', 
        # fmt='g'   # Alternative: General format (auto-detect int/float)
        xticklabels=class_labels,
        yticklabels=class_labels,
        cmap='Blues' 
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', pad=20)  
    
    # Save if path given - make folder if needed
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  
            os.makedirs(save_dir, exist_ok=True)  
        
        plt.savefig(save_path, bbox_inches='tight')  
        print(f'Saved confusion matrix to {save_path}')
    
    # Show after saving to prevent blank windows
    if show:
        plt.show()
    
    plt.close()

def plot_class_metrics(metrics_dict,class_names, save_path=None, show=True):
    """Plot precision/recall/F1 per class as grouped bars.
    """
    
    classes = sorted(metrics_dict.keys(), key=lambda x: int(x.split('_')[1]))
    labels = class_names or [f'Class {c.split("_")[1]}' for c in classes]
    metrics = ['precision', 'recall', 'f1']  
    
    # Setup figure before plotting
    plt.figure(figsize=(12, 6))
    plt.xticks(x_indices + bar_width, labels) 
    bar_width = 0.25
    x_indices = np.arange(len(classes))
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[cls][metric] for cls in classes]
        plt.bar(
            x_indices + i * bar_width,
            values, 
            bar_width,
            label=metric.capitalize() 
        )
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)  
    plt.xticks(x_indices + bar_width, [f'Class {c.split("_")[1]}' for c in classes])
    plt.legend()  
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  
            os.makedirs(save_dir, exist_ok=True)  
        
        plt.savefig(save_path, bbox_inches='tight')  # bbox stops text cutoff
        print(f'Saved class matrix to {save_path}')
        
    if show:
        plt.show()
    
    plt.close()  
    
# plot_confusion_matrix(cm, ['cat', 'dog'], save_path='results/cm.png')
# plot_class_metrics(metrics, save_path='results/class_metrics.pdf', show=False)