# Graph-Enhanced EEG-to-Text Decoding

A PyTorch implementation of **Graph-Enhanced EEG-to-Text Decoding: A Spatio-Temporal Relational Embedding Framework for Brain Signal Translation**.

This repository implements a novel graph-enhanced framework that explicitly models relational information in brain signals for decoding natural language directly from EEG recordings.

## Overview

The framework consists of three main components:

1. **Spectro-Topographic Relational Graphs (STRG)**: Constructs graphs that jointly encode static electrode topology and dynamic inter-channel functional connectivity
2. **Spatio-Temporal Relational Embeddings (STRE)**: Generates graph-aware representations using Graph Attention Networks (GAT) and Transformer encoders
3. **Transformer Decoder**: Generates natural language outputs from the STRE embeddings

## Features

- **Graph-based EEG representation**: Explicitly models spatial and functional relationships among electrodes
- **Multi-frequency analysis**: Processes delta, theta, alpha, beta, and gamma frequency bands
- **End-to-end training**: Supports composite loss with cross-entropy, graph smoothness, and contrastive alignment
- **Comprehensive evaluation**: Includes BLEU, ROUGE, and BERTScore metrics
- **ZuCo dataset support**: Preprocessing utilities for ZuCo v1.0 and v2.0 datasets

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Graph-Enhanced-EEG-to-Text-Generation.git
cd Graph-Enhanced-EEG-to-Text-Generation

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for metrics)
python -c "import nltk; nltk.download('punkt')"
```

<!-- ### Training

Train the model on the ZuCo dataset:

```bash
python train.py \
    --config config/config.yaml \
    --data_dir data \
    --checkpoint_dir checkpoints
```

Key training parameters can be modified in `config/config.yaml`:

- Model architecture (GAT layers, Transformer layers, dimensions)
- Training hyperparameters (learning rate, batch size, epochs)
- Loss weights (smoothness regularization, contrastive alignment) -->

<!-- ### Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data \
    --split test \
    --output evaluation_results.json
```

The evaluation script computes:
- BLEU scores (1-4)
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore (precision, recall, F1) -->

<!-- ### Inference

Generate text from EEG signals:

```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --eeg_file path/to/eeg_data.mat \
    --output generated_text.txt
```

Supported EEG file formats:
- MATLAB (.mat)
- CSV (.csv)
- NumPy (.npy) -->

<!-- ## Configuration

Edit `config/config.yaml` to customize:

### Model Parameters

```yaml
model:
  num_channels: 64              # Number of EEG channels
  num_frequency_bands: 5        # delta, theta, alpha, beta, gamma
  strg:
    alpha: 0.5                  # Spatial topology weight
    beta: 0.5                   # Functional connectivity weight
  stre:
    graph_embed_dim: 256        # Graph embedding dimension
    num_gat_layers: 2           # Number of GAT layers
    num_gat_heads: 4            # Number of attention heads in GAT
```

### Training Parameters

```yaml
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 1e-4
  lambda_smooth: 0.1            # Graph smoothness weight
  lambda_contrastive: 0.2       # Contrastive alignment weight
```

## Model Architecture

### STRG Construction

The Spectro-Topographic Relational Graph (STRG) encodes:
- **Nodes**: Electrode-frequency band pairs with band power features
- **Edges**: 
  - Static spatial adjacency based on electrode topology (10-20 system)
  - Dynamic functional connectivity using Pearson correlation

### STRE Generation

Spatio-Temporal Relational Embeddings are generated through:
1. **Graph encoding**: Multi-layer GAT processes STRG nodes
2. **Graph-level readout**: Attention-based aggregation of node embeddings
3. **Temporal modeling**: Transformer encoder captures temporal dependencies

### Decoder

Autoregressive Transformer decoder with:
- Masked self-attention for causal generation
- Cross-attention to STRE embeddings
- Composite loss (cross-entropy + smoothness + contrastive)

## Results

The model achieves:
- **BLEU-4**: 10.5 (16% relative improvement over baselines)
- **ROUGE-1-F**: 34.5
- **BERTScore-F**: 57.0

See the paper for detailed results and ablation studies.

## Citation

If you use this code, please cite:

```bibtex
@article{hippocampus2026graph,
  title={Graph-Enhanced EEG-to-Text Decoding: A Spatio-Temporal Relational Embedding Framework for Brain Signal Translation},
  author={Hippocampus, Antiquus S. and Cerebro, Natalia and Amygdale, Amelie P. and Ren, Ji Q. and LeNet, Yevgeny},
  journal={ICLR},
  year={2026}
} -->
```

## License

See LICENSE file for details.

## Acknowledgments

- ZuCo dataset by Hollenstein et al.
- PyTorch and Transformers libraries
- Graph Attention Networks (Velickovic et al., 2018)

## Contact

For questions or issues, please open a GitHub issue.