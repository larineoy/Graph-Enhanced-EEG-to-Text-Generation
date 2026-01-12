"""
Visualize ablation study results as bar chart with error bars
Shows component importance with mean ± std
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

from utils.statistics import compute_statistics, format_metric_with_std


def plot_ablation_results(
    results_dict: Dict[str, Dict],
    output_path: str,
    metric: str = 'bleu_4',
    figsize: tuple = (10, 6),
    show_error_bars: bool = True,
    significance_markers: Optional[Dict[str, str]] = None
):
    """
    Plot ablation study results as bar chart
    
    Args:
        results_dict: Dictionary mapping method_name -> {'mean': float, 'std': float, ...}
                     or method_name -> list of values (will compute stats)
        output_path: Path to save figure
        metric: Metric name to plot (e.g., 'bleu_4', 'rougeL_F')
        figsize: Figure size
        show_error_bars: Whether to show error bars
        significance_markers: Optional dict mapping method_name -> significance marker ('*', '**', '***')
    """
    # Process results - compute statistics if needed
    processed_results = {}
    for method_name, data in results_dict.items():
        if isinstance(data, dict):
            if 'mean' in data:
                processed_results[method_name] = data
            else:
                # List of values - compute stats
                stats = compute_statistics([{metric: v} for v in data])[metric]
                processed_results[method_name] = stats
        elif isinstance(data, list):
            # List of values - compute stats
            stats = compute_statistics([{metric: v} for v in data])[metric]
            processed_results[method_name] = stats
        else:
            # Single value
            processed_results[method_name] = {'mean': data, 'std': 0.0}
    
    # Extract data
    methods = list(processed_results.keys())
    means = [processed_results[m].get('mean', 0.0) for m in methods]
    stds = [processed_results[m].get('std', 0.0) for m in methods]
    
    # Convert to percentage if metric is BLEU/ROUGE and values are < 1
    is_percentage = any(m < 1 for m in means) and ('bleu' in metric.lower() or 'rouge' in metric.lower())
    if is_percentage:
        means = [m * 100 for m in means]
        stds = [s * 100 for s in stds]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme
    colors = ['#C73E1D', '#F18F01', '#F4A261', '#E9C46A', '#6A994E', '#2A9D8F', '#2E86AB']
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]
    
    # Plot bars
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds if show_error_bars else None,
                  capsize=5, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        label = f'{mean:.2f}' if std == 0 else format_metric_with_std(mean, std, decimals=2)
        # Add significance marker if available
        if significance_markers and methods[i] in significance_markers:
            label += f' {significance_markers[methods[i]]}'
        ax.text(i, mean + std + (max(means) * 0.02), label,
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize axes
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    metric_label = metric.upper().replace('_', '-')
    if is_percentage:
        metric_label += ' (%)'
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Study: {metric_label}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(bottom=0, top=max(means) * 1.3 if max(means) > 0 else 1)
    
    # Add legend for significance if markers present
    if significance_markers:
        legend_text = []
        if any('***' in m for m in significance_markers.values()):
            legend_text.append("*** p < 0.001")
        if any('**' in m for m in significance_markers.values()):
            legend_text.append("** p < 0.01")
        if any('*' in m for m in significance_markers.values()):
            legend_text.append("* p < 0.05")
        if legend_text:
            ax.text(0.98, 0.98, '\n'.join(legend_text), transform=ax.transAxes,
                   ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ablation results to {output_path}")


def load_ablation_results(results_dir: str, metric: str = 'bleu_4') -> Dict[str, Dict]:
    """
    Load ablation results from directory
    
    Args:
        results_dir: Directory containing result JSON files
        metric: Metric to extract
    
    Returns:
        Dictionary mapping method_name -> {'mean': float, 'std': float}
    """
    results = {}
    
    # Look for JSON files
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract method name from filename
            method_name = os.path.basename(json_file).replace('.json', '').replace('_results', '')
            
            # Handle different JSON formats
            if isinstance(data, list):
                # List of results (multi-seed)
                values = [r.get(metric, 0.0) for r in data if isinstance(r, dict)]
                if len(values) > 0:
                    from utils.statistics import compute_statistics
                    stats = compute_statistics([{metric: v} for v in values])[metric]
                    results[method_name] = stats
            elif isinstance(data, dict):
                if 'statistics' in data and metric in data['statistics']:
                    # Multi-seed aggregated format
                    results[method_name] = data['statistics'][metric]
                elif metric in data:
                    # Single result
                    results[method_name] = {'mean': data[metric], 'std': 0.0}
                elif 'all_results' in data:
                    # Multi-seed format
                    values = [r.get(metric, 0.0) for r in data['all_results'] if isinstance(r, dict)]
                    if len(values) > 0:
                        from utils.statistics import compute_statistics
                        stats = compute_statistics([{metric: v} for v in values])[metric]
                        results[method_name] = stats
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Visualize ablation study results')
    parser.add_argument('--results_dir', type=str, default='ablation_results',
                       help='Directory containing ablation result JSON files')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Single JSON file with all results (alternative to results_dir)')
    parser.add_argument('--metric', type=str, default='bleu_4',
                       choices=['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 
                               'rouge1_F', 'rougeL_F', 'bertscore_f1'],
                       help='Metric to visualize')
    parser.add_argument('--output', type=str, default='checkpoints/ablation_results.png',
                       help='Path to save figure')
    parser.add_argument('--no_error_bars', action='store_true',
                       help='Hide error bars')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUALIZING ABLATION RESULTS")
    print("="*70)
    
    # Load results
    if args.results_file:
        print(f"\nLoading results from {args.results_file}...")
        with open(args.results_file, 'r') as f:
            data = json.load(f)
        
        # Extract results for each method
        if isinstance(data, dict):
            results = {}
            for method_name, method_data in data.items():
                if isinstance(method_data, dict) and 'mean' in method_data:
                    results[method_name] = method_data
                elif isinstance(method_data, list):
                    from utils.statistics import compute_statistics
                    stats = compute_statistics([{args.metric: v} for v in method_data])[args.metric]
                    results[method_name] = stats
        else:
            print("Error: Invalid results file format")
            return
    else:
        print(f"\nLoading results from {args.results_dir}...")
        results = load_ablation_results(args.results_dir, args.metric)
    
    if len(results) == 0:
        print("Error: No results found")
        return
    
    print(f"✓ Loaded {len(results)} methods:")
    for method_name, stats in results.items():
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 0.0)
        print(f"  {method_name}: {format_metric_with_std(mean, std)}")
    
    # Plot
    print(f"\nGenerating bar chart for {args.metric}...")
    plot_ablation_results(
        results,
        args.output,
        metric=args.metric,
        show_error_bars=not args.no_error_bars
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
