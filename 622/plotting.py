"""
plotting.py
===========
Visualization for Vega Flow experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from typing import Optional


STYLE = {
    'opt':  {'color': '#2196F3', 'label': 'Optimal Strategy'},
    'hist': {'color': '#FF5722', 'label': 'Historical Strategy'},
    'real': {'color': '#212121', 'label': 'Real Flow'},
    'sim':  {'color': '#4CAF50', 'label': 'Simulated (OU)', 'alpha': 0.3},
}


def plot_riccati_coefficients(coefs, params, save_path: str = None):
    """Plot the Riccati ODE solution coefficients A, B, C, f, g, h over time."""
    t = coefs.t_grid
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Vega Riccati ODE Coefficients', fontsize=14, fontweight='bold')

    pairs = [
        (coefs.A, 'A(t)', 'Value fn X² coef'),
        (coefs.B, 'B(t)', 'Value fn XY coef'),
        (coefs.C, 'C(t)', 'Value fn Y² coef'),
        (coefs.f, 'f(t)', 'Feedback: inventory X'),
        (coefs.g, 'g(t)', 'Feedback: impact Y'),
        (coefs.h, 'h(t)', 'Feedback: flow Z'),
    ]

    for ax, (arr, name, desc) in zip(axes.flat, pairs):
        ax.plot(t, arr, color='#1565C0', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_title(f'{name}: {desc}', fontsize=10)
        ax.set_xlabel('Time t')
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experiment1_flow_replication(diag: dict, save_path: str = None):
    """
    Experiment 1: Real vega flow vs simulated OU flow comparison.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- ACF comparison ---
    ax1 = fig.add_subplot(gs[0, :2])
    lags = diag['acf_lags']
    ax1.plot(lags, diag['acf_real'], **STYLE['real'], linewidth=2, marker='o', markersize=4)
    ax1.plot(lags, diag['acf_sim_mean'], color=STYLE['sim']['color'],
             label='Simulated (OU) mean', linewidth=2)
    ax1.fill_between(lags,
                     diag['acf_sim_mean'] - 2*diag['acf_sim_std'],
                     diag['acf_sim_mean'] + 2*diag['acf_sim_std'],
                     color=STYLE['sim']['color'], alpha=0.2, label='Simulated ±2σ')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title('Autocorrelation Function: Real vs Simulated Vega Flow')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Moments table ---
    ax_m = fig.add_subplot(gs[0, 2])
    ax_m.axis('off')
    m = diag['moments']
    rows = [
        ['Metric', 'Real', 'Simulated'],
        ['Mean', f"{m['mean_real']:.3f}", f"{m['mean_sim']:.3f}"],
        ['Std',  f"{m['std_real']:.3f}",  f"{m['std_sim']:.3f}"],
        ['Skew', f"{m['skew_real']:.3f}", f"{m['skew_sim']:.3f}"],
        ['KS stat', f"{diag['ks_stat']:.3f}", ''],
        ['KS p-val', f"{diag['ks_pval']:.3f}", ''],
    ]
    tbl = ax_m.table(rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for j in range(3):
        tbl[0, j].set_facecolor('#1565C0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax_m.set_title('Moment Comparison')

    # --- Distribution comparison ---
    ax2 = fig.add_subplot(gs[1, 0])
    # (placeholder - in production, use real Z_real and Z_sim arrays)
    ax2.set_title('Marginal Distribution\n(requires Z arrays)')
    ax2.set_xlabel('Vega Flow Z')
    ax2.text(0.5, 0.5, 'Pass Z_real & Z_sim arrays\nfor histogram comparison',
             ha='center', va='center', transform=ax2.transAxes, fontsize=9, color='gray')
    ax2.grid(True, alpha=0.3)

    # --- Momentum metric ---
    ax3 = fig.add_subplot(gs[1, 1])
    mom_real = diag['momentum_real']
    mom_sim_mean = diag['momentum_sim_mean']
    mom_sim_std = diag['momentum_sim_std']

    ax3.bar(['Real', 'Sim Mean'], [mom_real, mom_sim_mean],
            color=[STYLE['real']['color'], STYLE['sim']['color']],
            alpha=0.8)
    ax3.errorbar(['Sim Mean'], [mom_sim_mean], yerr=[2*mom_sim_std],
                 color='black', fmt='none', capsize=6)
    ax3.set_title('Momentum Metric\n|Z_T| / TV(Z)')
    ax3.set_ylabel('Metric value')
    ax3.grid(True, alpha=0.3, axis='y')

    # --- KS test result panel ---
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ks_text = (
        f"KS Test Result\n\n"
        f"KS Statistic: {diag['ks_stat']:.4f}\n"
        f"p-value: {diag['ks_pval']:.4f}\n\n"
        f"{'✓ Fail to reject H0' if diag['ks_pval'] > 0.05 else '✗ Reject H0'}\n"
        f"(H0: distributions match)\n\n"
        f"Momentum:\n"
        f"  Real: {mom_real:.3f}\n"
        f"  Sim: {mom_sim_mean:.3f} ± {mom_sim_std:.3f}"
    )
    ax4.text(0.1, 0.9, ks_text, transform=ax4.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Experiment 1: Calibrated Flow vs Real Vega Flow', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experiment2_cost_comparison(df_comp, summary: dict, save_path: str = None):
    """
    Experiment 2: Optimal strategy cost vs historical strategy cost.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 2: Optimal Strategy vs Historical Strategy', fontsize=14, fontweight='bold')

    # --- Cost time series ---
    ax1 = axes[0, 0]
    x = np.arange(len(df_comp))
    ax1.plot(x, df_comp['cost_opt'],  **STYLE['opt'],  linewidth=1.5)
    ax1.plot(x, df_comp['cost_hist'], **STYLE['hist'], linewidth=1.5)
    ax1.set_title('Daily Cost: Optimal vs Historical')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Cost (bps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Daily savings ---
    ax2 = axes[0, 1]
    savings = df_comp['saving_bps']
    colors = ['#2196F3' if s > 0 else '#FF5722' for s in savings]
    ax2.bar(x, savings, color=colors, alpha=0.7)
    ax2.axhline(savings.mean(), color='k', linestyle='--', linewidth=1.5,
                label=f'Mean saving = {savings.mean():.2f} bps')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_title('Daily Cost Saving (Hist − Opt)')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Saving (bps)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Cost ratio distribution ---
    ax3 = axes[1, 0]
    ratio = df_comp['cost_ratio']
    ax3.hist(ratio, bins=30, color='#1565C0', alpha=0.7, edgecolor='white')
    ax3.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Ratio = 1 (break even)')
    ax3.axvline(ratio.mean(), color='k', linestyle='-', linewidth=1.5,
                label=f'Mean = {ratio.mean():.3f}')
    ax3.set_title('Distribution of Cost Ratio (Opt/Hist)')
    ax3.set_xlabel('Cost Ratio')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Summary stats ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    pct = summary['pct_days_opt_cheaper']
    stat_text = (
        f"Statistical Summary\n"
        f"{'─'*32}\n"
        f"Mean saving:      {summary['mean_saving_bps']:+.3f} bps\n"
        f"Median saving:    {summary['median_saving_bps']:+.3f} bps\n"
        f"Std of saving:    {summary['std_saving_bps']:.3f} bps\n"
        f"Opt cheaper (%):  {pct:.1f}%\n"
        f"Mean cost ratio:  {summary['mean_cost_ratio']:.4f}\n"
        f"{'─'*32}\n"
        f"Paired t-test:\n"
        f"  t-stat:  {summary['t_stat']:.3f}\n"
        f"  p-value: {summary['p_value']:.4f}\n"
        f"  {'Significant (α=5%)' if summary['p_value'] < 0.05 else 'Not significant'}"
    )
    ax4.text(0.05, 0.95, stat_text, transform=ax4.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_optimal_path(result: dict, Z_path: np.ndarray, t_grid: np.ndarray,
                       save_path: str = None):
    """Plot a single optimal strategy path: inventory, impact, q*, flow."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Optimal Vega Strategy: Single Path', fontsize=13)

    axes[0,0].plot(t_grid, result['X'], color='#1565C0', linewidth=2)
    axes[0,0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0,0].set_title('Vega Inventory X_t')
    axes[0,0].set_xlabel('Time'); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t_grid, result['Y'], color='#C62828', linewidth=2)
    axes[0,1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0,1].set_title('Vol Impact State Y_t')
    axes[0,1].set_xlabel('Time'); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(t_grid, result['q'], color='#2E7D32', linewidth=2)
    axes[1,0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1,0].set_title('Optimal Trading Speed q*_t')
    axes[1,0].set_xlabel('Time'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(t_grid, Z_path, color='#6A1B9A', linewidth=1.5)
    axes[1,1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1,1].set_title('Vega Flow Z_t (OU)')
    axes[1,1].set_xlabel('Time'); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
