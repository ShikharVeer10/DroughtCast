import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize':12,
    'axes.titlesize':14,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'figure.titlesize':16
})

def plot_temporal_attention_heatmaps(attention_weights: np.ndarray,sample_indices:list=None,save_path:str="outputs/attention_heatmap.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.squeeze(-1)
        
    if sample_indices is None:
        sample_indices = list(range(min(5, attention_weights.shape[0])))
        
    subset_weights = attention_weights[sample_indices]
    
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(
        subset_weights, 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Attention Weight (alpha)'},
        xticklabels=[f"T - {6-i}" for i in range(6)],
        yticklabels=[f"Sample {idx+1}" for idx in sample_indices]
    )
    
    plt.title("Temporal Attention Weight Distribution Across Historical Lookback", pad=15, fontweight='bold')
    plt.xlabel("Historical Time Steps", labelpad=10)
    plt.ylabel("Evaluation Samples", labelpad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Attention heatmap successfully saved to {save_path}")


def plot_multi_horizon_forecasts(predictions: np.ndarray, targets: np.ndarray, horizon_name: str = "SPEI-3", save_path: str = "outputs/forecast_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    
    step_labels = ["T + 1 Month", "T + 2 Months", "T + 3 Months", "T + 4 Months"]
    
    for step_idx in range(4):
        ax = axes[step_idx]
        pred_step = predictions[:, step_idx]
        targ_step = targets[:, step_idx]
        
        ax.plot(targ_step, label="Observed (Ground Truth)", color="#2c3e50", linewidth=1.5, alpha=0.9)
        ax.plot(pred_step, label=f"Predicted ({horizon_name})", color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.9)
        
        ax.set_title(f"Forecast Horizon: {step_labels[step_idx]}", fontweight='bold')
        ax.set_xlabel("Time Index")
        ax.set_ylabel("SPEI Index Value")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)
        
    fig.suptitle(f"Out-of-Sample Multi-Horizon Verification: {horizon_name}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Forecast comparison plot for {horizon_name} saved to {save_path}")