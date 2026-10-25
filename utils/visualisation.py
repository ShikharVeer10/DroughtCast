import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, pearsonr

def plot_hexbin_extreme_calibration(y_true_dict, y_pred_dict, save_path="plots/fig4_hexbin_calibration.pdf"):
    """
    Upgrades image_badddf.png into a 3-panel log-density Hexbin plot with extreme-tail shading
    and hydrological efficiency metrics (R2, RMSE, KGE, Extreme-Tail RMSE).
    y_true_dict / y_pred_dict: {'SPEI-3 (Agri)': arr, 'SPEI-6 (Hydro)': arr, 'SPEI-12 (Groundwater)': arr}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    cmaps = ['Greens', 'Purples', 'Reds']
    
    for idx, (title, y_true) in enumerate(y_true_dict.items()):
        ax = axes[idx]
        y_pred = y_pred_dict[title]
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        
        # 1. Shaded Extreme Drought Zone (SPEI <= -1.5)
        ax.axvspan(-3.2, -1.5, ymin=0, ymax=(-1.5 - (-3.2)) / 6.4, color='#ffcccc', alpha=0.35, zorder=0, label='Severe/Extreme Regime')
        
        # 2. Log-density Hexbin
        hb = ax.hexbin(yt, yp, gridsize=45, cmap=cmaps[idx], bins='log', mincnt=1, edgecolors='none', zorder=2)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('$\log_{10}(\mathrm{Count})$')
        
        # 3. 1:1 Line & Metrics
        ax.plot([-3.2, 3.2], [-3.2, 3.2], 'k--', lw=1.3, zorder=3, label='1:1 Identity')
        
        r2 = 1.0 - np.sum((yt - yp)**2) / np.sum((yt - np.mean(yt))**2)
        rmse = np.sqrt(np.mean((yt - yp)**2))
        
        # Kling-Gupta Efficiency (KGE)
        r, _ = pearsonr(yt, yp)
        alpha = np.std(yp) / np.std(yt)
        beta = np.mean(yp) / (np.mean(yt) + 1e-8)
        kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
        
        # Extreme Tail RMSE (SPEI <= -1.5)
        ext_mask = yt <= -1.5
        rmse_ext = np.sqrt(np.mean((yt[ext_mask] - yp[ext_mask])**2)) if np.any(ext_mask) else np.nan
        
        stats_text = (
            f"$R^2 = {r2:.3f}$\n"
            f"$\mathrm{{KGE}} = {kge:.3f}$\n"
            f"$\mathrm{{RMSE}} = {rmse:.3f}$\n"
            f"$\mathrm{{RMSE}}_{{\\leq -1.5}} = {rmse_ext:.3f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Observed SPEI (Ground Truth)')
        if idx == 0:
            ax.set_ylabel('DroughtCast Predicted SPEI')
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multihorizon_seasonal_attention(attn_3x6, seasonal_12x6, save_path="plots/fig6_xai_attention.pdf"):
    """
    Upgrades Figure 9 in image_baddbe.jpg into a 2-panel physical explainability plot:
    Panel A: Horizon-Specific Attention (SPEI-3, SPEI-6, SPEI-12 x Lags t-6..t-1)
    Panel B: Seasonal Attention Shift across all 12 calendar months (Jan-Dec x Lags t-6..t-1)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1, 1.3]})
    lags = ['t-6', 't-5', 't-4', 't-3', 't-2', 't-1']
    
    # Panel A: Grouped Bar or Heatmap across Drought Horizons
    horizons = ['SPEI-3\n(Agri)', 'SPEI-6\n(Hydro)', 'SPEI-12\n(Groundwater)']
    im1 = ax1.imshow(attn_3x6 * 100, cmap='magma', aspect='auto', vmin=5, vmax=45)
    ax1.set_xticks(range(6))
    ax1.set_xticklabels(lags)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(horizons, fontweight='bold')
    ax1.set_xlabel('Historical Lag Window (Months)')
    ax1.set_title('(a) Cross-Horizon Memory Allocation (%)', fontweight='bold')
    
    for i in range(3):
        for j in range(6):
            val = attn_3x6[i, j] * 100
            color = 'white' if val < 25 else 'black'
            ax1.text(j, i, f"{val:.1f}%", ha='center', va='center', color=color, fontsize=8.5, fontweight='bold')
            
    # Panel B: 12-Month Seasonal Attention Shift
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    im2 = ax2.imshow(seasonal_12x6 * 100, cmap='viridis', aspect='auto', vmin=5, vmax=45)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(lags)
    ax2.set_yticks(range(12))
    ax2.set_yticklabels(months)
    ax2.set_xlabel('Historical Lag Window (Months)')
    ax2.set_ylabel('Target Forecast Month')
    ax2.set_title('(b) Seasonal Attention Shift (Monsoon Memory Dynamics)', fontweight='bold')
    
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.025, pad=0.03)
    cbar.set_label('Attention Weight (%)', fontweight='bold')
    plt.savefig(save_path)
    plt.close()