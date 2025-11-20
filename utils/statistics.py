"""
Statistical analysis utilities
Includes multi-seed evaluation, significance testing, and error bars
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import json
import os


def compute_statistics(results_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and standard deviation across multiple runs
    
    Args:
        results_list: List of result dictionaries, each from one run
        
    Returns:
        statistics: Dictionary with 'mean' and 'std' for each metric
    """
    if len(results_list) == 0:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for results in results_list:
        all_metrics.update(results.keys())
    
    statistics = {}
    for metric in all_metrics:
        values = [results.get(metric, np.nan) for results in results_list]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            statistics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),  # Sample std (Bessel's correction)
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'n_runs': len(values)
            }
    
    return statistics


def format_metric_with_std(mean: float, std: float, decimals: int = 2) -> str:
    """
    Format metric as mean ± std
    
    Args:
        mean: Mean value
        std: Standard deviation
        decimals: Number of decimal places
        
    Returns:
        formatted_string: "mean ± std"
    """
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def paired_t_test(results_a: List[float], results_b: List[float]) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of results
    
    Args:
        results_a: List of metric values for method A
        results_b: List of metric values for method B
        
    Returns:
        test_results: Dictionary with t-statistic, p-value, and significance
    """
    if len(results_a) != len(results_b):
        raise ValueError("Results must be paired (same length)")
    
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(results_a, results_b)
    
    # Determine significance
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'ns'
    
    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'significance': significance,
        'mean_diff': float(np.mean(results_a - results_b))
    }


def wilcoxon_test(results_a: List[float], results_b: List[float]) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    
    Args:
        results_a: List of metric values for method A
        results_b: List of metric values for method B
        
    Returns:
        test_results: Dictionary with statistic, p-value, and significance
    """
    if len(results_a) != len(results_b):
        raise ValueError("Results must be paired (same length)")
    
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    # Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(results_a, results_b)
    
    # Determine significance
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'ns'
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significance': significance,
        'mean_diff': float(np.mean(results_a - results_b))
    }


def compute_significance_matrix(all_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute significance tests between all pairs of methods for each metric
    
    Args:
        all_results: Dictionary mapping method_name -> list of result dicts
        
    Returns:
        significance_matrix: Nested dict: metric -> (method_a, method_b) -> test_results
    """
    # Get all metrics
    all_metrics = set()
    for method_results in all_results.values():
        for result in method_results:
            all_metrics.update(result.keys())
    
    significance_matrix = {}
    
    method_names = list(all_results.keys())
    
    for metric in all_metrics:
        significance_matrix[metric] = {}
        
        # Extract metric values for each method
        method_metric_values = {}
        for method_name in method_names:
            values = [r.get(metric, np.nan) for r in all_results[method_name]]
            values = [v for v in values if not np.isnan(v)]
            if len(values) > 0:
                method_metric_values[method_name] = values
        
        # Perform pairwise tests
        for i, method_a in enumerate(method_names):
            if method_a not in method_metric_values:
                continue
            for j, method_b in enumerate(method_names):
                if i >= j or method_b not in method_metric_values:
                    continue
                
                values_a = method_metric_values[method_a]
                values_b = method_metric_values[method_b]
                
                # Pad to same length if needed (for unpaired case)
                min_len = min(len(values_a), len(values_b))
                if len(values_a) == len(values_b):
                    # Paired test
                    test_result = paired_t_test(values_a[:min_len], values_b[:min_len])
                else:
                    # Unpaired test
                    t_stat, p_val = stats.ttest_ind(values_a[:min_len], values_b[:min_len])
                    test_result = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significance': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns',
                        'mean_diff': float(np.mean(values_a[:min_len]) - np.mean(values_b[:min_len]))
                    }
                
                significance_matrix[metric][(method_a, method_b)] = test_result
    
    return significance_matrix


def save_results_table_with_errors(
    all_results: Dict[str, List[Dict[str, float]]],
    output_path: str,
    significance_matrix: Optional[Dict] = None,
    format: str = 'latex'
):
    """
    Save results table with mean ± std and significance markers
    
    Args:
        all_results: Dictionary mapping method_name -> list of result dicts
        output_path: Path to save table
        significance_matrix: Optional significance test results
        format: Output format ('latex', 'csv', 'markdown')
    """
    # Compute statistics for each method
    method_stats = {}
    for method_name, results_list in all_results.items():
        method_stats[method_name] = compute_statistics(results_list)
    
    # Get all metrics
    all_metrics = set()
    for stats_dict in method_stats.values():
        all_metrics.update(stats_dict.keys())
    
    # Create table
    table_data = []
    for method_name in sorted(all_results.keys()):
        row = {'Method': method_name}
        for metric in sorted(all_metrics):
            if metric in method_stats[method_name]:
                stats = method_stats[method_name][metric]
                row[metric] = format_metric_with_std(stats['mean'], stats['std'])
            else:
                row[metric] = 'N/A'
        
        # Add significance markers if available
        if significance_matrix:
            for metric in sorted(all_metrics):
                if metric in significance_matrix:
                    # Check if this method is significantly better than baseline
                    # (simplified - would need baseline method name)
                    pass
        
        table_data.append(row)
    
    # Save as DataFrame
    df = pd.DataFrame(table_data)
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'latex':
        df.to_latex(output_path, index=False, escape=False, float_format='%.2f')
    elif format == 'markdown':
        df.to_markdown(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Saved results table to {output_path}")


def aggregate_multi_seed_results(results_dir: str, output_path: str) -> Dict:
    """
    Aggregate results from multiple seed runs
    
    Args:
        results_dir: Directory containing result JSON files from different seeds
        output_path: Path to save aggregated results
        
    Returns:
        aggregated_results: Statistics across all seeds
    """
    import glob
    
    result_files = glob.glob(os.path.join(results_dir, '*_seed_*.json'))
    
    if len(result_files) == 0:
        print(f"No seed-specific result files found in {results_dir}")
        return {}
    
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    # Compute statistics
    statistics = compute_statistics(all_results)
    
    # Save aggregated results
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Aggregated {len(all_results)} seed runs, saved to {output_path}")
    
    return statistics

