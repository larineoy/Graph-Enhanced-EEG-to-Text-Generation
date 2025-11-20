"""
Utility modules for Graph-Enhanced EEG-to-Text Decoding
"""

from .metrics import evaluate_predictions, compute_bleu, compute_rouge, compute_bertscore
from .losses import CompositeLoss
from .visualization import (
    save_adjacency_heatmap,
    save_spatial_functional_comparison,
    save_graph_evolution,
    visualize_learned_graphs_from_checkpoint
)
from .statistics import (
    compute_statistics,
    format_metric_with_std,
    paired_t_test,
    wilcoxon_test,
    compute_significance_matrix,
    save_results_table_with_errors,
    aggregate_multi_seed_results
)
from .ablation import AblationModelFactory, get_ablation_loss_weights
from .cross_subject import CrossSubjectEvaluator
from .baselines import (
    create_shuffled_channel_baseline,
    create_shuffled_time_baseline,
    create_random_gaussian_baseline,
    create_random_uniform_baseline
)
from .sensitivity import (
    generate_hyperparameter_grid,
    generate_loss_weight_grid,
    generate_architecture_grid
)

__all__ = [
    'evaluate_predictions',
    'compute_bleu',
    'compute_rouge',
    'compute_bertscore',
    'CompositeLoss',
    'save_adjacency_heatmap',
    'save_spatial_functional_comparison',
    'save_graph_evolution',
    'visualize_learned_graphs_from_checkpoint',
    'compute_statistics',
    'format_metric_with_std',
    'paired_t_test',
    'wilcoxon_test',
    'compute_significance_matrix',
    'save_results_table_with_errors',
    'aggregate_multi_seed_results',
    'AblationModelFactory',
    'get_ablation_loss_weights',
    'CrossSubjectEvaluator',
    'create_shuffled_channel_baseline',
    'create_shuffled_time_baseline',
    'create_random_gaussian_baseline',
    'create_random_uniform_baseline',
    'generate_hyperparameter_grid',
    'generate_loss_weight_grid',
    'generate_architecture_grid'
]
