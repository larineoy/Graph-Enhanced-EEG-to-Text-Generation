# Visualization Guide for Paper

This guide outlines recommended visualizations for your Graph-Enhanced EEG-to-Text Generation paper.

## 🎯 Essential Visualizations (Must Have)

### 1. **Architecture Diagram** ⭐⭐⭐
**Purpose**: Show overall model structure
**Status**: ❌ Not implemented (need to create)
**What to show**:
- STRG → STRE → Decoder pipeline
- Input: EEG frequency bands
- Output: Text tokens
- Key components: Graph construction, GAT layers, Transformer decoder

**Tools**: Use draw.io, TikZ, or Python (matplotlib + graphviz)

---

### 2. **STRG Adjacency Matrix (Block-Diagonal Structure)** ⭐⭐⭐
**Purpose**: Demonstrate learned graph structure
**Status**: ✅ Already implemented
**Function**: `save_strg_adjacency_with_bands()`
**What it shows**:
- Block-diagonal structure (frequency bands)
- Edge weights (spatial + functional)
- Clear frequency band annotations

**Usage**:
```python
python visualization/generate_visualizations.py --checkpoint checkpoints/best_model.pt
# Generates: checkpoints/visualizations/adjacency_heatmap.png
```

**Paper placement**: Main results figure, architecture section

---

### 3. **Spatial vs Functional vs Combined Adjacency** ⭐⭐⭐
**Purpose**: Show contribution of each component
**Status**: ✅ Already implemented
**Function**: `save_spatial_functional_comparison()`
**What it shows**:
- Spatial adjacency (topology-based)
- Functional connectivity (correlation-based)
- Combined (weighted sum)

**Usage**: Part of `visualize_strg_comprehensive()`

**Paper placement**: Method section, ablation discussion

---

### 4. **Training Curves** ⭐⭐⭐
**Purpose**: Show model convergence
**Status**: ⚠️ Data exists but not visualized
**What to show**:
- Train/Val loss over epochs
- Individual loss components (CE, contrastive, smoothness)
- Validation metrics (BLEU, ROUGE) over epochs

**Data source**: `checkpoints/training_log.json`

**Implementation needed**: Create plotting script

---

### 5. **Example Predictions** ⭐⭐⭐
**Purpose**: Qualitative results
**Status**: ❌ Not implemented
**What to show**:
- Input EEG signal (sample)
- Ground truth text
- Model prediction
- Highlight correct/incorrect parts

**Implementation needed**: Create visualization script

---

## 🔬 Important Visualizations (Should Have)

### 6. **Frequency-Specific Connectivity** ⭐⭐
**Purpose**: Show connectivity patterns per band
**Status**: ✅ Already implemented
**Function**: `save_frequency_specific_connectivity()`
**What it shows**:
- Functional connectivity for delta, theta, alpha, beta, gamma
- Different patterns across bands

**Paper placement**: Results section, supplementary

---

### 7. **Bandpower Topographic Maps** ⭐⭐
**Purpose**: Neuroscientific interpretation
**Status**: ✅ Already implemented
**Function**: `save_bandpower_topographic_maps()`
**What it shows**:
- Bandpower distribution across electrodes
- Per frequency band
- Spatial patterns on scalp

**Paper placement**: Results section, interpretation

---

### 8. **Ablation Study Results** ⭐⭐
**Purpose**: Component importance
**Status**: ⚠️ Data exists but not visualized
**What to show**:
- Bar chart: BLEU-4 for each ablation
- Error bars (std across seeds)
- Significance markers

**Data source**: Ablation study results
**Implementation needed**: Create bar chart with error bars

---

### 9. **Graph Network Structure** ⭐⭐
**Purpose**: Visualize actual graph topology
**Status**: ✅ Already implemented (if networkx available)
**Function**: `save_graph_network_structure()`
**What it shows**:
- Networkx graph visualization
- Node colors by frequency band
- Edge weights as thickness/color

**Paper placement**: Supplementary, method illustration

---

## 📊 Supporting Visualizations (Nice to Have)

### 10. **Performance Comparison Bar Chart** ⭐
**Purpose**: Compare methods
**Status**: ⚠️ Can use `save_results_table_with_errors()` but needs bar chart
**What to show**:
- Bar chart: BLEU-4, ROUGE-L-F for each method
- Error bars (mean ± std)
- Baseline, ablations, full model

**Implementation needed**: Convert table to bar chart

---

### 11. **Cross-Subject Performance** ⭐
**Purpose**: Generalization analysis
**Status**: ❌ Not implemented
**What to show**:
- Box plot or bar chart per subject
- LOSO cross-validation results
- Subject-specific performance

**Implementation needed**: Create from cross-subject results

---

### 12. **Attention Heatmaps** ⭐
**Purpose**: Interpretability
**Status**: ❌ Not implemented
**What to show**:
- Decoder attention weights
- Which EEG regions attend to which words
- Temporal attention patterns

**Implementation needed**: Extract attention from decoder

---

### 13. **Graph Evolution Over Training** ⭐
**Purpose**: Show learning dynamics
**Status**: ✅ Already implemented
**Function**: `save_graph_evolution()`
**What it shows**:
- Adjacency matrices at different epochs
- How graph structure changes during training

**Paper placement**: Supplementary, training analysis

---

### 14. **Electrode Importance Analysis** ⭐
**Purpose**: Identify important brain regions
**Status**: ❌ Not implemented
**What to show**:
- Feature importance per electrode
- Which channels contribute most to predictions
- Language-related regions (if known)

**Implementation needed**: Gradient-based or ablation-based importance

---

### 15. **Frequency Band Contribution** ⭐
**Purpose**: Understand frequency importance
**Status**: ❌ Not implemented
**What to show**:
- Ablation: remove each frequency band
- Performance drop per band
- Which bands are most critical

**Implementation needed**: Frequency-specific ablation study

---

## 📝 Recommended Paper Figure Structure

### **Figure 1: Architecture Overview**
- Model pipeline diagram
- STRG → STRE → Decoder flow
- Input/output examples

### **Figure 2: Learned Graph Structure**
- STRG adjacency matrix (block-diagonal)
- Spatial vs Functional comparison
- Frequency-specific connectivity

### **Figure 3: Training and Results**
- Training curves (loss, metrics)
- Ablation study bar chart
- Performance comparison

### **Figure 4: Qualitative Examples**
- Example predictions (good/bad cases)
- Attention visualizations
- Error analysis

### **Figure 5: Neuroscientific Analysis** (Optional)
- Topographic maps
- Electrode importance
- Cross-subject performance

---

## 🛠️ Implementation Priority

### **High Priority** (Implement first):
1. ✅ STRG adjacency matrix - **Already done**
2. ✅ Spatial/Functional comparison - **Already done**
3. ⚠️ Training curves - **Data exists, need plotting**
4. ❌ Architecture diagram - **Need to create**
5. ❌ Example predictions - **Need to create**

### **Medium Priority**:
6. ✅ Frequency-specific connectivity - **Already done**
7. ✅ Topographic maps - **Already done**
8. ⚠️ Ablation bar chart - **Data exists, need visualization**
9. ✅ Graph network structure - **Already done**

### **Low Priority** (Nice to have):
10. Attention heatmaps
11. Cross-subject visualization
12. Electrode importance
13. Frequency band ablation

---

## 🚀 Quick Start: Generate Existing Visualizations

```bash
# Generate all STRG visualizations from trained model
python visualization/generate_visualizations.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data \
    --output_dir checkpoints/visualizations
```

This creates:
- `adjacency_heatmap.png` - Main adjacency matrix
- `strg_components.png` - Spatial/Functional/Combined comparison
- `frequency_specific_connectivity.png` - Per-band connectivity
- `bandpower_topography.png` - Topographic maps
- `graph_network_structure.png` - Network visualization
- `strg_adjacency_bands.png` - Annotated block-diagonal structure

---

## 📋 Missing Visualizations to Implement

### 1. **Training Curves Plotter**
```python
# visualize_training_curves.py
import json
import matplotlib.pyplot as plt
from utils.statistics import format_metric_with_std

def plot_training_curves(log_path, output_path):
    with open(log_path) as f:
        log = json.load(f)
    
    epochs = [e['epoch'] for e in log]
    train_loss = [e['train_loss'] for e in log]
    val_loss = [e['val_loss'] for e in log]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(epochs, train_loss, label='Train', marker='o')
    axes[0].plot(epochs, val_loss, label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics (if available)
    # ... plot BLEU, ROUGE over epochs
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

### 2. **Example Predictions Visualizer**
```python
# visualize_examples.py
def visualize_predictions(model, dataloader, tokenizer, num_examples=5):
    # Show: EEG sample → Ground truth → Prediction
    # Highlight differences
```

### 3. **Ablation Bar Chart**
```python
# visualize_ablation_results.py
def plot_ablation_bars(ablation_results, output_path):
    # Bar chart with error bars
    # Significance markers
```

---

## 💡 Pro Tips

1. **Use high DPI**: All figures should be 300 DPI for publication
2. **Consistent color scheme**: Use same colors across figures
3. **Clear labels**: Include axis labels, legends, titles
4. **Frequency band annotations**: Always label delta/theta/alpha/beta/gamma
5. **Error bars**: Include std for all quantitative results
6. **Significance markers**: Add *, **, *** for statistical significance

---

## Summary

**Already Available** (✅):
- STRG adjacency matrices
- Spatial/Functional comparison
- Frequency-specific connectivity
- Topographic maps
- Graph network structure

**Need to Create** (❌):
- Architecture diagram
- Training curves
- Example predictions
- Ablation bar charts
- Attention visualizations

**Data Exists, Need Plotting** (⚠️):
- Training curves (from training_log.json)
- Ablation results (from ablation study)
