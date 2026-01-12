# Visualization Scripts Guide

This guide explains how to use all the visualization scripts created for your paper.

## 📊 Available Visualization Scripts

### 1. **Training Curves** (`visualize_training_curves.py`)
**Purpose**: Plot training and validation loss/metrics over epochs

**Usage**:
```bash
python visualization/visualize_training_curves.py \
    --log checkpoints/training_log.json \
    --output checkpoints/training_curves.png
```

**What it shows**:
- Train/Val loss curves
- BLEU scores over epochs
- ROUGE and BERTScore over epochs

**Requirements**: `checkpoints/training_log.json` (automatically created during training)

---

### 2. **Example Predictions** (`visualize_examples.py`)
**Purpose**: Show qualitative examples (EEG → Ground Truth → Prediction)

**Usage**:
```bash
python visualization/visualize_examples.py \
    --checkpoint checkpoints/best_model.pt \
    --config config/config.yaml \
    --data_dir data \
    --split val \
    --num_examples 5 \
    --output checkpoints/example_predictions.png
```

**What it shows**:
- Ground truth text
- Model predictions
- Word overlap percentage
- Visual indicators (good/partial/poor match)

**Requirements**: Trained model checkpoint

---

### 3. **Ablation Study Results** (`visualize_ablation_results.py`)
**Purpose**: Bar chart comparing ablation study variants

**Usage**:
```bash
# From directory of result files
python visualization/visualize_ablation_results.py \
    --results_dir ablation_results \
    --metric bleu_4 \
    --output checkpoints/ablation_results.png

# From single JSON file
python visualization/visualize_ablation_results.py \
    --results_file ablation_results/all_results.json \
    --metric bleu_4 \
    --output checkpoints/ablation_results.png
```

**What it shows**:
- Bar chart with error bars (mean ± std)
- Significance markers (*, **, ***)
- Multiple metrics support

**Requirements**: Ablation study results (JSON format)

---

### 4. **Performance Comparison** (`visualize_performance_comparison.py`)
**Purpose**: Compare all methods (baselines, ablations, full model)

**Usage**:
```bash
python visualization/visualize_performance_comparison.py \
    --results_file results/all_methods.json \
    --metrics bleu_4 rougeL_F bertscore_f1 \
    --baseline baseline_method \
    --output checkpoints/performance_comparison.png
```

**What it shows**:
- Side-by-side bar charts for multiple metrics
- Error bars for each method
- Significance markers (if baseline provided)

**Requirements**: JSON file with results for all methods

**JSON Format**:
```json
{
  "baseline": {"bleu_4": {"mean": 0.1, "std": 0.02}, ...},
  "no_graph": {"bleu_4": {"mean": 0.15, "std": 0.03}, ...},
  "full_model": {"bleu_4": {"mean": 0.25, "std": 0.04}, ...}
}
```

---

### 5. **Cross-Subject Performance** (`visualize_cross_subject.py`)
**Purpose**: Show generalization across different subjects

**Usage**:
```bash
# From directory
python visualization/visualize_cross_subject.py \
    --results_dir cross_subject_results \
    --metric bleu_4 \
    --plot_type box \
    --output checkpoints/cross_subject_performance.png

# From single JSON file
python visualization/visualize_cross_subject.py \
    --results_file cross_subject_results/all_subjects.json \
    --metric bleu_4 \
    --plot_type box \
    --output checkpoints/cross_subject_performance.png
```

**What it shows**:
- Performance per subject (box plot, bar plot, or violin plot)
- Overall mean/std across subjects
- Subject-specific variability

**Plot Types**: `box`, `bar`, or `violin`

**Requirements**: Cross-subject evaluation results

---

### 6. **Attention Heatmaps** (`visualize_attention.py`)
**Purpose**: Visualize decoder attention patterns (simplified)

**Usage**:
```bash
python visualization/visualize_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --config config/config.yaml \
    --data_dir data \
    --split val \
    --num_examples 5 \
    --output checkpoints/attention_heatmaps.png
```

**What it shows**:
- Token probability distribution (Top-K)
- Attention-like patterns (using logits as proxy)

**Note**: This shows token probability distributions as a proxy for attention. For true attention weights, the decoder would need to be modified to return attention matrices.

**Requirements**: Trained model checkpoint

---

### 7. **Architecture Diagram** (`visualize_architecture.py`)
**Purpose**: Generate model architecture diagram

**Usage**:
```bash
python visualization/visualize_architecture.py \
    --output checkpoints/architecture_diagram.png \
    --dpi 300
```

**What it shows**:
- STRG → STRE → Decoder pipeline
- Input/output flow
- Component labels and colors

**Requirements**: None (standalone script)

---

## 🚀 Quick Start: Generate All Visualizations

Create a script to run all visualizations at once:

```bash
#!/bin/bash
# generate_all_visualizations.sh

echo "Generating architecture diagram..."
python visualization/visualize_architecture.py --output checkpoints/architecture_diagram.png

echo "Generating training curves..."
python visualization/visualize_training_curves.py --log checkpoints/training_log.json --output checkpoints/training_curves.png

echo "Generating example predictions..."
python visualization/visualize_examples.py --checkpoint checkpoints/best_model.pt --config config/config.yaml --data_dir data --split val --num_examples 5 --output checkpoints/example_predictions.png

echo "Generating attention heatmaps..."
python visualization/visualize_attention.py --checkpoint checkpoints/best_model.pt --config config/config.yaml --data_dir data --split val --num_examples 5 --output checkpoints/attention_heatmaps.png

# Ablation and comparison plots (require result files)
if [ -d "ablation_results" ]; then
    echo "Generating ablation results..."
    python visualization/visualize_ablation_results.py --results_dir ablation_results --metric bleu_4 --output checkpoints/ablation_results.png
fi

if [ -f "results/all_methods.json" ]; then
    echo "Generating performance comparison..."
    python visualization/visualize_performance_comparison.py --results_file results/all_methods.json --metrics bleu_4 rougeL_F --output checkpoints/performance_comparison.png
fi

if [ -d "cross_subject_results" ]; then
    echo "Generating cross-subject performance..."
    python visualization/visualize_cross_subject.py --results_dir cross_subject_results --metric bleu_4 --plot_type box --output checkpoints/cross_subject_performance.png
fi

echo "All visualizations generated!"
```

---

## 📋 Data Requirements

### **Available Immediately** (from existing training):
- ✅ Training curves → `checkpoints/training_log.json`
- ✅ Architecture diagram → No data needed
- ✅ Example predictions → Model checkpoint
- ✅ Attention heatmaps → Model checkpoint

### **Requires Additional Evaluation**:
- ⚠️ Ablation results → Run ablation studies
- ⚠️ Performance comparison → Run baseline evaluations
- ⚠️ Cross-subject performance → Run cross-subject evaluation

---

## 📝 Output Files

All visualizations are saved as high-resolution PNG files (300 DPI) ready for paper inclusion:

- `checkpoints/architecture_diagram.png` - Architecture overview
- `checkpoints/training_curves.png` - Training/validation curves
- `checkpoints/example_predictions.png` - Qualitative examples
- `checkpoints/attention_heatmaps.png` - Attention patterns
- `checkpoints/ablation_results.png` - Ablation study
- `checkpoints/performance_comparison.png` - Method comparison
- `checkpoints/cross_subject_performance.png` - Cross-subject analysis

---

## 🎨 Figure Quality

All scripts generate publication-ready figures:
- **Resolution**: 300 DPI (configurable)
- **Format**: PNG (vector formats can be added if needed)
- **Size**: Optimized for paper inclusion
- **Colors**: Consistent color scheme across all plots
- **Labels**: Clear axis labels, legends, titles

---

## 💡 Tips

1. **Training Curves**: Already available from your training run
2. **Example Predictions**: Run on validation set for best examples
3. **Ablation Study**: Run ablation studies first to generate data
4. **Cross-Subject**: Run cross-subject evaluation first
5. **Attention**: Simplified visualization (true attention requires decoder modification)

---

## 🔧 Troubleshooting

**Issue**: "Training log not found"
- **Solution**: Ensure `train.py` has been run and saved `training_log.json`

**Issue**: "No results found" (ablation/cross-subject)
- **Solution**: Run the corresponding evaluation scripts first

**Issue**: "Model checkpoint not found"
- **Solution**: Train a model first, or specify correct checkpoint path

**Issue**: "Tokenizer not available"
- **Solution**: Install transformers: `pip install transformers`

---

## 📚 Related Documentation

- See `VISUALIZATION_GUIDE.md` for recommendations on which visualizations to use
- See `EVALUATION_GUIDE.md` for how to generate evaluation data
- See `README.md` for overall project documentation
