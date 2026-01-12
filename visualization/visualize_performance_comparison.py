"""
Visualize performance comparison across methods
Bar chart comparing main model, baselines, and ablations
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import sys
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.statistics import compute_statistics, format_metric_with_std, paired_t_test


def plot_performance_comparison(
    all_results: Dict[str, Dict],
    output_path: str,
    metrics: List[str] = ['bleu_4', 'rougeL_F', 'bertscore_f1'],
    figsize: tuple = (14, 6),
    baseline_method: Optional[str] = None
):
    """
    Plot performance comparison across multiple methods and metrics
    
    Args:
        all_results: Dictionary mapping method_name -> results_dict
                    results_dict should have statistics for each metric
        output_path: Path to save figure
        metrics: List of metrics to plot
        figsize: Figure size
        baseline_method: Name of baseline method for significance testing
    """
    # Extract methods
    methods = list(all_results.keys())
    
    # Process results - ensure we have statistics
    processed_results = {}
    for method_name, data in all_results.items():
        processed_results[method_name] = {}
        for metric in metrics:
            if metric in data:
                if isinstance(data[metric], dict) and 'mean' in data[metric]:
                    processed_results[method_name][metric] = data[metric]
                elif isinstance(data[metric], list):
                    stats = compute_statistics([{metric: v} for v in data[metric]])[metric]
                    processed_results[method_name][metric] = stats
                else:
                    processed_results[method_name][metric] = {'mean': data[metric], 'std': 0.0}
            else:
                processed_results[method_name][metric] = {'mean': 0.0, 'std': 0.0}
    
    # Compute significance markers if baseline provided
    significance_markers = {}
    if baseline_method and baseline_method in processed_results:
        baseline_stats = processed_results[baseline_method]
        for method_name in methods:
            if method_name == baseline_method:
                continue
            for metric in metrics:
                key = f"{method_name}_{metric}"
                if method_name in processed_results and metric in processed_results[method_name]:
                    # Simple comparison: check if mean is significantly different
                    method_stats = processed_results[method_name][metric]
                    baseline_mean = baseline_stats[metric].get('mean', 0.0)
                    method_mean = method_stats.get('mean', 0.0)
                    baseline_std = baseline_stats[metric].get('std', 0.0)
                    method_std = method_stats.get('std', 0.0)
                    
                    # Rough significance check (simplified)
                    diff = method_mean - baseline_mean
                    combined_std = np.sqrt(baseline_std**2 + method_std**2) if baseline_std > 0 and method_std > 0 else abs(diff)
                    if combined_std > 0:
                        z_score = diff / combined_std
                        if abs(z_score) > 2.58:
                            significance_markers[key] = '***'
                        elif abs(z_score) > 1.96:
                            significance_markers[key] = '**'
                        elif abs(z_score) > 1.65:
                            significance_markers[key] = '*'
    
    # Create figure with subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Color scheme
    colors = ['#C73E1D', '#F18F01', '#F4A261', '#E9C46A', '#6A994E', '#2A9D8F', '#2E86AB']
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Extract data for this metric
        means = []
        stds = []
        method_colors = []
        
        for i, method_name in enumerate(methods):
            stats = processed_results[method_name][metric]
            mean = stats.get('mean', 0.0)
            std = stats.get('std', 0.0)
            
            # Convert to percentage if needed
            is_percentage = mean < 1 and ('bleu' in metric.lower() or 'rouge' in metric.lower())
            if is_percentage:
                mean = mean * 100
                std = std * 100
            
            means.append(mean)
            stds.append(std)
            method_colors.append(colors[i % len(colors)])
        
        # Plot bars
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=method_colors,
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            label = format_metric_with_std(mean, std, decimals=2)
            key = f"{methods[i]}_{metric}"
            if key in significance_markers:
                label += f' {significance_markers[key]}'
            ax.text(i, mean + std + (max(means) * 0.03), label,
                   ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)
        
        # Customize
        metric_label = metric.upper().replace('_', '-')
        if is_percentage:
            metric_label += ' (%)'
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_title(metric_label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(bottom=0, top=max(means) * 1.4 if max(means) > 0 else 1)
    
    # Add overall title
    fig.suptitle('Performance Comparison Across Methods', fontsize=14, fontweight='bold', y=1.02)
    
    # Add significance legend
    if significance_markers:
        legend_text = "*** p < 0.001, ** p < 0.01, * p < 0.05"
        fig.text(0.5, 0.01, legend_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved performance comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize performance comparison')
    parser.add_argument('--results_file', type=str, required=True,
                       help='JSON file with results for all methods')
    parser.add_argument('--metrics', nargs='+', 
                       default=['bleu_4', 'rougeL_F', 'bertscore_f1'],
                       help='Metrics to plot')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Baseline method name for significance testing')
    parser.add_argument('--output', type=str, default='checkpoints/performance_comparison.png',
                       help='Path to save figure')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUALIZING PERFORMANCE COMPARISON")
    print("="*70)
    
    # Load results
    print(f"\nLoading results from {args.results_file}...")
    with open(args.results_file, 'r') as f:
        data = json.load(f)
    
    # Process data format
    if isinstance(data, dict):
        # Assume format: {method_name: {metric: value or {mean, std}}}
        all_results = data
    else:
        print("Error: Invalid results file format")
        return
    
    print(f"✓ Loaded {len(all_results)} methods")
    print(f"✓ Metrics: {args.metrics}")
    
    # Plot
    plot_performance_comparison(
        all_results,
        args.output,
        metrics=args.metrics,
        baseline_method=args.baseline
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
