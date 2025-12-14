#!/usr/bin/env python3
"""
analyze_results.py - Ithemal Training Results Analysis

Analyzes training output and generates visualizations similar to the original paper.
Supports Haswell, Skylake, and Ivy Bridge architectures.

Usage:
    python analyze_results.py --arch ivb --results_dir ./saved/bhive_ivb/1765596981/
    python analyze_results.py --arch all --base_dir ./saved/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def parse_validation_results(filepath):
    """
    Parse validation_results.txt file.
    Format: predicted_value,ground_truth_value (one per line)
    Last lines contain loss and statistics.
    
    Returns:
        DataFrame with columns ['predicted', 'actual']
        float: final loss value
    """
    predicted = []
    actual = []
    final_loss = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check for loss line
            if line.startswith('loss -'):
                final_loss = float(line.split('-')[1].strip())
                continue
            
            # Skip if line doesn't contain comma
            if ',' not in line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    pred = float(parts[0])
                    act = float(parts[1])
                    # Filter out obviously invalid data
                    if pred > 0 and act > 0:
                        predicted.append(pred)
                        actual.append(act)
                except ValueError:
                    continue
    
    df = pd.DataFrame({
        'predicted': predicted,
        'actual': actual
    })
    
    return df, final_loss


def parse_loss_report(filepath):
    """
    Parse loss_report.log file.
    Format: epoch | elapsed_time | loss | batch_size (tab-separated)
    
    Returns:
        DataFrame with columns ['epoch', 'time', 'loss', 'batch_size']
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    epoch = int(parts[0])
                    time = float(parts[1])
                    loss = float(parts[2])
                    batch_size = int(parts[3])
                    data.append({
                        'epoch': epoch,
                        'time': time,
                        'loss': loss,
                        'batch_size': batch_size
                    })
                except ValueError:
                    continue
    
    return pd.DataFrame(data)


def calculate_metrics(df):
    """
    Calculate accuracy metrics.
    
    Returns:
        dict with pearson, spearman, mape, mae, rmse, and filtered pearson values
    """
    pred = df['predicted'].values
    actual = df['actual'].values
    
    # Pearson correlation (all data)
    pearson_r, pearson_p = stats.pearsonr(pred, actual)
    
    # Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(pred, actual)
    
    # Filtered Pearson correlations (removing outliers)
    # Filter 1: actual < 1000 cycles
    mask_1000 = actual < 1000
    if np.sum(mask_1000) > 10:
        pearson_r_1000, _ = stats.pearsonr(pred[mask_1000], actual[mask_1000])
        n_filtered_1000 = np.sum(mask_1000)
    else:
        pearson_r_1000 = np.nan
        n_filtered_1000 = 0
    
    # Filter 2: actual < 500 cycles
    mask_500 = actual < 500
    if np.sum(mask_500) > 10:
        pearson_r_500, _ = stats.pearsonr(pred[mask_500], actual[mask_500])
        n_filtered_500 = np.sum(mask_500)
    else:
        pearson_r_500 = np.nan
        n_filtered_500 = 0
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = actual > 0
    mape = np.mean(np.abs(pred[mask] - actual[mask]) / actual[mask]) * 100
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred - actual))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    
    # Accuracy within X%
    relative_error = np.abs(pred[mask] - actual[mask]) / actual[mask]
    within_10 = np.mean(relative_error <= 0.10) * 100
    within_20 = np.mean(relative_error <= 0.20) * 100
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'pearson_r_1000': pearson_r_1000,
        'n_filtered_1000': n_filtered_1000,
        'pearson_r_500': pearson_r_500,
        'n_filtered_500': n_filtered_500,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'within_10_pct': within_10,
        'within_20_pct': within_20,
        'n_samples': len(df)
    }


def plot_accuracy_heatmap(df, arch_name, output_dir, max_cycles=1000):
    """
    Plot 2D histogram heatmap of predicted vs actual throughput.
    Replicates Paper Figure 2 & 6.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filter data to focus on smaller throughputs
    mask = (df['actual'] < max_cycles) & (df['predicted'] < max_cycles)
    filtered_df = df[mask]
    
    # Create 2D histogram with log scale
    h = ax.hist2d(
        filtered_df['actual'], 
        filtered_df['predicted'],
        bins=100,
        cmap='viridis',
        norm=plt.matplotlib.colors.LogNorm(),
        cmin=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count (log scale)')
    
    # Add diagonal line (perfect prediction)
    lims = [0, max_cycles]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction (y=x)')
    
    # Labels and title
    ax.set_xlabel('Measured Throughput (CPU Cycles)')
    ax.set_ylabel('Predicted Throughput (CPU Cycles)')
    ax.set_title(f'Ithemal Prediction Accuracy - {arch_name.upper()}\n(n={len(filtered_df):,} samples)')
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_dir / f'{arch_name}_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{arch_name}_accuracy_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {arch_name}_accuracy_heatmap.png/pdf")


def plot_accuracy_heatmap_full(df, arch_name, output_dir):
    """
    Plot full range heatmap (including larger throughputs).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use 99th percentile as max to avoid outliers
    max_val = max(df['actual'].quantile(0.99), df['predicted'].quantile(0.99))
    
    # Create 2D histogram with log scale
    h = ax.hist2d(
        df['actual'], 
        df['predicted'],
        bins=100,
        cmap='viridis',
        norm=plt.matplotlib.colors.LogNorm(),
        cmin=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count (log scale)')
    
    # Add diagonal line
    lims = [0, max_val]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction (y=x)')
    
    ax.set_xlabel('Measured Throughput (CPU Cycles)')
    ax.set_ylabel('Predicted Throughput (CPU Cycles)')
    ax.set_title(f'Ithemal Prediction Accuracy (Full Range) - {arch_name.upper()}')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{arch_name}_accuracy_heatmap_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {arch_name}_accuracy_heatmap_full.png")


def plot_error_by_throughput(df, arch_name, output_dir, bin_size=50):
    """
    Plot average error by throughput range.
    Replicates Paper Figure 3 & 7.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter out zeros and compute errors
    mask = df['actual'] > 0
    df_valid = df[mask].copy()
    df_valid['abs_error'] = np.abs(df_valid['predicted'] - df_valid['actual'])
    df_valid['pct_error'] = df_valid['abs_error'] / df_valid['actual'] * 100
    
    # Create bins
    max_throughput = min(df_valid['actual'].quantile(0.95), 1000)
    bins = np.arange(0, max_throughput + bin_size, bin_size)
    df_valid['bin'] = pd.cut(df_valid['actual'], bins=bins, labels=bins[:-1])
    
    # Aggregate by bin
    agg_df = df_valid.groupby('bin').agg({
        'abs_error': ['mean', 'std', 'count'],
        'pct_error': ['mean', 'std']
    }).reset_index()
    agg_df.columns = ['bin', 'mae', 'mae_std', 'count', 'mape', 'mape_std']
    agg_df['bin'] = agg_df['bin'].astype(float)
    agg_df = agg_df[agg_df['count'] >= 10]  # Filter bins with too few samples
    
    # Plot 1: Absolute Error
    ax1 = axes[0]
    ax1.errorbar(agg_df['bin'], agg_df['mae'], yerr=agg_df['mae_std']/np.sqrt(agg_df['count']),
                 fmt='o-', capsize=3, capthick=1, markersize=5, color='steelblue')
    ax1.set_xlabel('Measured Throughput (CPU Cycles)')
    ax1.set_ylabel('Mean Absolute Error (Cycles)')
    ax1.set_title(f'Absolute Error by Throughput - {arch_name.upper()}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentage Error
    ax2 = axes[1]
    ax2.errorbar(agg_df['bin'], agg_df['mape'], yerr=agg_df['mape_std']/np.sqrt(agg_df['count']),
                 fmt='o-', capsize=3, capthick=1, markersize=5, color='darkorange')
    ax2.set_xlabel('Measured Throughput (CPU Cycles)')
    ax2.set_ylabel('Mean Absolute Percentage Error (%)')
    ax2.set_title(f'Percentage Error by Throughput - {arch_name.upper()}')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{arch_name}_error_by_throughput.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{arch_name}_error_by_throughput.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {arch_name}_error_by_throughput.png/pdf")


def plot_learning_curves(loss_df, arch_name, output_dir):
    """
    Plot training loss over time/epochs.
    Replicates Paper Figure 5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss over time
    ax1 = axes[0]
    ax1.plot(loss_df['time'] / 60, loss_df['loss'], alpha=0.3, linewidth=0.5, color='blue')
    
    # Add smoothed line (rolling average)
    window = min(100, len(loss_df) // 10)
    if window > 1:
        smoothed = loss_df['loss'].rolling(window=window, min_periods=1).mean()
        ax1.plot(loss_df['time'] / 60, smoothed, linewidth=2, color='darkblue', label=f'Smoothed (window={window})')
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Training Loss over Time - {arch_name.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss by epoch
    ax2 = axes[1]
    epoch_loss = loss_df.groupby('epoch')['loss'].agg(['mean', 'min', 'max']).reset_index()
    ax2.fill_between(epoch_loss['epoch'], epoch_loss['min'], epoch_loss['max'], alpha=0.3, color='steelblue')
    ax2.plot(epoch_loss['epoch'], epoch_loss['mean'], linewidth=2, color='darkblue', label='Mean loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title(f'Training Loss by Epoch - {arch_name.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{arch_name}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{arch_name}_learning_curves.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {arch_name}_learning_curves.png/pdf")


def plot_distribution_comparison(df, arch_name, output_dir):
    """
    Plot histogram comparing predicted vs actual distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Focus on reasonable range
    max_val = min(df['actual'].quantile(0.95), 1000)
    
    # Plot 1: Overlaid histograms
    ax1 = axes[0]
    ax1.hist(df['actual'], bins=100, alpha=0.5, label='Measured', color='blue', 
             range=(0, max_val), density=True)
    ax1.hist(df['predicted'], bins=100, alpha=0.5, label='Predicted', color='orange',
             range=(0, max_val), density=True)
    ax1.set_xlabel('Throughput (CPU Cycles)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution Comparison - {arch_name.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    ax2 = axes[1]
    mask = df['actual'] > 0
    errors = (df.loc[mask, 'predicted'] - df.loc[mask, 'actual']) / df.loc[mask, 'actual'] * 100
    errors_clipped = errors.clip(-100, 100)  # Clip extreme values for visualization
    
    ax2.hist(errors_clipped, bins=100, alpha=0.7, color='steelblue', edgecolor='white')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.axvline(x=errors.median(), color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {errors.median():.1f}%')
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Error Distribution - {arch_name.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{arch_name}_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {arch_name}_distributions.png")


def plot_scatter_sample(df, arch_name, output_dir, sample_size=5000):
    """
    Plot scatter plot with a sample of points for clearer visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample data for clearer visualization
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    # Focus on reasonable range
    max_val = min(sample_df['actual'].quantile(0.99), 1500)
    mask = (sample_df['actual'] < max_val) & (sample_df['predicted'] < max_val)
    plot_df = sample_df[mask]
    
    # Calculate error for coloring
    errors = np.abs(plot_df['predicted'] - plot_df['actual']) / plot_df['actual'] * 100
    
    scatter = ax.scatter(
        plot_df['actual'], 
        plot_df['predicted'],
        c=errors,
        cmap='RdYlGn_r',
        alpha=0.5,
        s=10,
        vmin=0,
        vmax=50
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Percentage Error (%)')
    
    # Add diagonal line
    lims = [0, max_val]
    ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Measured Throughput (CPU Cycles)')
    ax.set_ylabel('Predicted Throughput (CPU Cycles)')
    ax.set_title(f'Prediction Scatter Plot - {arch_name.upper()} (n={len(plot_df):,} samples)')
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{arch_name}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {arch_name}_scatter.png")


def analyze_architecture(arch_name, results_dir, output_dir):
    """
    Run full analysis for one architecture.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {arch_name.upper()}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}")
    
    # Parse validation results
    val_file = results_dir / 'validation_results.txt'
    if not val_file.exists():
        print(f"ERROR: {val_file} not found!")
        return None
    
    print(f"\nParsing validation results...")
    df, final_loss = parse_validation_results(val_file)
    print(f"  Loaded {len(df):,} samples")
    print(f"  Final loss: {final_loss:.6f}" if final_loss else "  Final loss: N/A")
    
    # Parse loss report
    loss_file = results_dir / 'loss_report.log'
    loss_df = None
    if loss_file.exists():
        print(f"\nParsing loss report...")
        loss_df = parse_loss_report(loss_file)
        print(f"  Loaded {len(loss_df):,} training steps")
        print(f"  Epochs: {loss_df['epoch'].min()} - {loss_df['epoch'].max()}")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    metrics = calculate_metrics(df)
    
    # Print metrics table
    print(f"\n{'─'*60}")
    print(f"METRICS SUMMARY - {arch_name.upper()}")
    print(f"{'─'*60}")
    print(f"  Samples:                    {metrics['n_samples']:,}")
    print(f"  Pearson Correlation (all):  {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  Pearson (<1000 cycles):     {metrics['pearson_r_1000']:.4f} (n={metrics['n_filtered_1000']:,})")
    print(f"  Pearson (<500 cycles):      {metrics['pearson_r_500']:.4f} (n={metrics['n_filtered_500']:,})")
    print(f"  Spearman Correlation:       {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"  MAPE:                       {metrics['mape']:.2f}%")
    print(f"  MAE:                        {metrics['mae']:.2f} cycles")
    print(f"  RMSE:                       {metrics['rmse']:.2f} cycles")
    print(f"  Within 10% error:           {metrics['within_10_pct']:.1f}%")
    print(f"  Within 20% error:           {metrics['within_20_pct']:.1f}%")
    print(f"{'─'*60}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    plot_accuracy_heatmap(df, arch_name, output_dir)
    plot_accuracy_heatmap_full(df, arch_name, output_dir)
    plot_error_by_throughput(df, arch_name, output_dir)
    plot_distribution_comparison(df, arch_name, output_dir)
    plot_scatter_sample(df, arch_name, output_dir)
    
    if loss_df is not None and len(loss_df) > 0:
        plot_learning_curves(loss_df, arch_name, output_dir)
    
    print(f"\n✅ Analysis complete for {arch_name.upper()}")
    
    return {
        'arch': arch_name,
        'metrics': metrics,
        'final_loss': final_loss
    }


def compare_architectures(results_list, output_dir):
    """
    Create comparison plots across architectures.
    """
    if len(results_list) < 2:
        return
    
    output_dir = Path(output_dir)
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    archs = [r['arch'].upper() for r in results_list]
    mapes = [r['metrics']['mape'] for r in results_list]
    pearsons = [r['metrics']['pearson_r'] for r in results_list]
    spearmans = [r['metrics']['spearman_r'] for r in results_list]
    
    colors = ['steelblue', 'darkorange', 'forestgreen'][:len(archs)]
    
    # MAPE comparison
    axes[0].bar(archs, mapes, color=colors)
    axes[0].set_ylabel('MAPE (%)')
    axes[0].set_title('Mean Absolute Percentage Error')
    for i, v in enumerate(mapes):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    # Pearson comparison
    axes[1].bar(archs, pearsons, color=colors)
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Pearson Correlation Coefficient')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(pearsons):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Spearman comparison
    axes[2].bar(archs, spearmans, color=colors)
    axes[2].set_ylabel('Spearman Correlation')
    axes[2].set_title('Spearman Rank Correlation')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(spearmans):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: comparison_metrics.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Analyze Ithemal training results')
    parser.add_argument('--arch', type=str, default='ivb',
                        help='Architecture to analyze (ivb, hsw, skl, or all)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to results directory (for single arch)')
    parser.add_argument('--base_dir', type=str, default='./saved',
                        help='Base directory containing all results')
    parser.add_argument('--output_dir', type=str, default='./plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_list = []
    
    if args.arch.lower() == 'all':
        # Process all architectures
        arch_configs = {
            'ivb': 'bhive_ivb',
            'hsw': 'bhive_hsw', 
            'skl': 'bhive_skl'
        }
        
        for arch, folder in arch_configs.items():
            base_path = Path(args.base_dir) / folder
            if base_path.exists():
                # Find the latest timestamp directory
                subdirs = [d for d in base_path.iterdir() if d.is_dir()]
                if subdirs:
                    latest = max(subdirs, key=lambda x: x.name)
                    result = analyze_architecture(arch, latest, output_dir)
                    if result:
                        results_list.append(result)
    else:
        # Process single architecture
        if args.results_dir:
            results_dir = args.results_dir
        else:
            # Try to find results directory
            arch_folder = f'bhive_{args.arch.lower()}'
            base_path = Path(args.base_dir) / arch_folder
            if base_path.exists():
                subdirs = [d for d in base_path.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = max(subdirs, key=lambda x: x.name)
                else:
                    results_dir = base_path
            else:
                results_dir = args.base_dir
        
        result = analyze_architecture(args.arch.lower(), results_dir, output_dir)
        if result:
            results_list.append(result)
    
    # Create comparison if multiple architectures
    if len(results_list) > 1:
        compare_architectures(results_list, output_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for r in results_list:
        print(f"\n{r['arch'].upper()}:")
        print(f"  MAPE: {r['metrics']['mape']:.2f}%")
        print(f"  Pearson (all):       {r['metrics']['pearson_r']:.4f}")
        print(f"  Pearson (<1000 cyc): {r['metrics']['pearson_r_1000']:.4f}")
        print(f"  Pearson (<500 cyc):  {r['metrics']['pearson_r_500']:.4f}")
        print(f"  Spearman:            {r['metrics']['spearman_r']:.4f}")
    
    print(f"\n✅ All plots saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()