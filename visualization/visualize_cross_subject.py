"""
Visualize cross-subject performance
Shows generalization across different subjects
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


def plot_cross_subject_performance(
    results_dict: Dict[str, Dict],
    output_path: str,
    metric: str = 'bleu_4',
    figsize: tuple = (12, 6),
    show_error_bars: bool = True,
    plot_type: str = 'box'  # 'box', 'bar', or 'violin'
):
    """
    Plot cross-subject performance
    
    Args:
        results_dict: Dictionary mapping subject_id -> {'mean': float, 'std': float, ...}
                     or subject_id -> list of values (will compute stats)
        output_path: Path to save figure
        metric: Metric name to plot
        figsize: Figure size
        show_error_bars: Whether to show error bars (for bar plot)
        plot_type: Type of plot ('box', 'bar', or 'violin')
    """
    # Process results - compute statistics if needed
    processed_results = {}
    all_values = []
    
    for subject_id, data in results_dict.items():
        if isinstance(data, dict):
            if 'mean' in data:
                processed_results[subject_id] = data
                if 'values' in data:
                    all_values.extend(data['values'])
            else:
                # List of values - compute stats
                stats = compute_statistics([{metric: v} for v in data])[metric]
                processed_results[subject_id] = stats
                all_values.extend(data)
        elif isinstance(data, list):
            # List of values - compute stats
            stats = compute_statistics([{metric: v} for v in data])[metric]
            processed_results[subject_id] = stats
            all_values.extend(data)
        else:
            # Single value
            processed_results[subject_id] = {'mean': data, 'std': 0.0}
    
    # Extract data
    subjects = sorted(processed_results.keys())
    means = [processed_results[s].get('mean', 0.0) for s in subjects]
    stds = [processed_results[s].get('std', 0.0) for s in subjects]
    
    # Convert to percentage if metric is BLEU/ROUGE and values are < 1
    is_percentage = any(m < 1 for m in means) and ('bleu' in metric.lower() or 'rouge' in metric.lower())
    if is_percentage:
        means = [m * 100 for m in means]
        stds = [s * 100 for s in stds]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == 'box':
        # Box plot - need raw values per subject
        data_for_box = []
        for subject_id in subjects:
            data = results_dict[subject_id]
            if isinstance(data, dict) and 'values' in data:
                values = data['values']
            elif isinstance(data, list):
                values = data
            else:
                # Create dummy values from mean/std
                mean = processed_results[subject_id].get('mean', 0.0)
                std = processed_results[subject_id].get('std', 0.0)
                # Generate sample values (simplified - ideally use real values)
                values = np.random.normal(mean, std, 10).tolist() if std > 0 else [mean]
            
            if is_percentage:
                values = [v * 100 if v < 1 else v for v in values]
            data_for_box.append(values)
        
        bp = ax.boxplot(data_for_box, labels=subjects, patch_artist=True, showmeans=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    elif plot_type == 'violin':
        # Violin plot - need raw values per subject
        data_for_violin = []
        for subject_id in subjects:
            data = results_dict[subject_id]
            if isinstance(data, dict) and 'values' in data:
                values = data['values']
            elif isinstance(data, list):
                values = data
            else:
                mean = processed_results[subject_id].get('mean', 0.0)
                std = processed_results[subject_id].get('std', 0.0)
                values = np.random.normal(mean, std, 10).tolist() if std > 0 else [mean]
            
            if is_percentage:
                values = [v * 100 if v < 1 else v for v in values]
            data_for_violin.append(values)
        
        parts = ax.violinplot(data_for_violin, positions=range(len(subjects)), showmeans=True, showmedians=True)
        
        # Color violins
        colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45, ha='right')
    
    else:  # bar plot (default)
        # Bar plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
        bars = ax.bar(range(len(subjects)), means, yerr=stds if show_error_bars else None,
                     capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            label = format_metric_with_std(mean, std, decimals=2)
            ax.text(i, mean + std + (max(means) * 0.02), label,
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=10)
    
    # Add overall statistics
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {overall_mean:.2f}')
    ax.axhline(y=overall_mean + overall_std, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=overall_mean - overall_std, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Customize axes
    ax.set_xlabel('Subject ID', fontsize=12, fontweight='bold')
    metric_label = metric.upper().replace('_', '-')
    if is_percentage:
        metric_label += ' (%)'
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Subject Performance: {metric_label}', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis limits
    y_min = min(means) - max(stds) * 1.5 if show_error_bars else min(means) * 0.9
    y_max = max(means) + max(stds) * 1.5 if show_error_bars else max(means) * 1.2
    ax.set_ylim(bottom=max(0, y_min), top=y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cross-subject performance to {output_path}")


def load_cross_subject_results(results_dir: str, metric: str = 'bleu_4') -> Dict[str, Dict]:
    """
    Load cross-subject results from directory
    
    Args:
        results_dir: Directory containing cross-subject result JSON files
        metric: Metric to extract
    
    Returns:
        Dictionary mapping subject_id -> {'mean': float, 'std': float}
    """
    results = {}
    
    # Look for JSON files
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract subject ID from filename (e.g., "subject_ZAB_results.json" -> "ZAB")
            filename = os.path.basename(json_file)
            if 'subject' in filename.lower():
                subject_id = filename.replace('subject_', '').replace('_results.json', '').replace('.json', '')
            else:
                subject_id = filename.replace('_results.json', '').replace('.json', '')
            
            # Handle different JSON formats
            if isinstance(data, list):
                values = [r.get(metric, 0.0) for r in data if isinstance(r, dict)]
                if len(values) > 0:
                    stats = compute_statistics([{metric: v} for v in values])[metric]
                    stats['values'] = values
                    results[subject_id] = stats
            elif isinstance(data, dict):
                if 'statistics' in data and metric in data['statistics']:
                    stats = data['statistics'][metric].copy()
                    if 'all_results' in data:
                        stats['values'] = [r.get(metric, 0.0) for r in data['all_results']]
                    results[subject_id] = stats
                elif metric in data:
                    results[subject_id] = {'mean': data[metric], 'std': 0.0}
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Visualize cross-subject performance')
    parser.add_argument('--results_dir', type=str, default='cross_subject_results',
                       help='Directory containing cross-subject result JSON files')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Single JSON file with all cross-subject results')
    parser.add_argument('--metric', type=str, default='bleu_4',
                       choices=['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 
                               'rouge1_F', 'rougeL_F', 'bertscore_f1'],
                       help='Metric to visualize')
    parser.add_argument('--output', type=str, default='checkpoints/cross_subject_performance.png',
                       help='Path to save figure')
    parser.add_argument('--plot_type', type=str, default='box',
                       choices=['box', 'bar', 'violin'],
                       help='Type of plot')
    parser.add_argument('--no_error_bars', action='store_true',
                       help='Hide error bars (for bar plot)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUALIZING CROSS-SUBJECT PERFORMANCE")
    print("="*70)
    
    # Load results
    if args.results_file:
        print(f"\nLoading results from {args.results_file}...")
        with open(args.results_file, 'r') as f:
            data = json.load(f)
        
        # Extract results for each subject
        if isinstance(data, dict):
            results = {}
            for subject_id, subject_data in data.items():
                if isinstance(subject_data, dict) and 'mean' in subject_data:
                    results[subject_id] = subject_data
                elif isinstance(subject_data, list):
                    stats = compute_statistics([{args.metric: v} for v in subject_data])[args.metric]
                    stats['values'] = subject_data
                    results[subject_id] = stats
        else:
            print("Error: Invalid results file format")
            return
    else:
        print(f"\nLoading results from {args.results_dir}...")
        results = load_cross_subject_results(args.results_dir, args.metric)
    
    if len(results) == 0:
        print("Error: No results found")
        return
    
    print(f"✓ Loaded {len(results)} subjects:")
    for subject_id, stats in sorted(results.items()):
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 0.0)
        print(f"  {subject_id}: {format_metric_with_std(mean, std)}")
    
    # Plot
    print(f"\nGenerating {args.plot_type} plot for {args.metric}...")
    plot_cross_subject_performance(
        results,
        args.output,
        metric=args.metric,
        show_error_bars=not args.no_error_bars,
        plot_type=args.plot_type
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
