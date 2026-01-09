# Supervised LDA for Co-Pathology Analysis

A Python implementation of supervised Latent Dirichlet Allocation (sLDA) for investigating co-pathology patterns in neurodegenerative diseases using regional gray matter atrophy data.

## Overview

This implementation adapts sLDA for continuous neuroimaging features:
- **Input**: Regional brain atrophy values (62 cortical regions)
- **Output**: Latent pathology patterns (topics) and diagnosis predictions
- **Use case**: Discover co-pathology in AD, PD, DLB, SVAD, and HC patients

## Files

| File | Description |
|------|-------------|
| `preprocessing.py` | Data loading and preparation utilities |
| `slda_model.py` | Core sLDA model implementation (PyMC) |
| `visualization.py` | Plotting functions for interpretation |
| `inference_new_subjects.py` | Utilities for inferring new subjects |
| `slda_copathology_example.ipynb` | Full training workflow example |
| `inference_example.ipynb` | New subject inference examples |

## Quick Start

### 1. Training a Model

```python
from preprocessing import load_wsev_data, prepare_slda_inputs
from slda_model import CoPathologySLDA

# Load data
df = load_wsev_data('/path/to/data.csv')
X, y, feature_names, dx_labels = prepare_slda_inputs(df)

# Train model
model = CoPathologySLDA(n_topics=4)
model.fit(X, y, n_samples=2000, tune=1000, chains=4)

# Extract results
topic_patterns = model.get_topic_patterns()        # K×62 matrix
patient_mixtures = model.get_patient_mixtures()    # N×K matrix
diagnosis_weights = model.get_diagnosis_weights()  # K×5 matrix
```

### 2. Inferring New Subjects

```python
from inference_new_subjects import infer_new_subject, print_inference_results

# New subject with 62 cortical features
X_new = new_subject_data[feature_names].values

# Infer diagnosis and topic mixture
results = infer_new_subject(
    model,
    X_new,
    feature_names,
    dx_labels,
    subject_id="PATIENT_001"
)

# Display results
print_inference_results(results)

# Access specific values
print(f"Predicted diagnosis: {results['predicted_diagnosis']}")
print(f"Diagnosis probabilities: {results['diagnosis_breakdown']}")
print(f"Topic mixture (co-pathology): {results['topic_mixture']}")
```

### 3. Batch Inference

```python
from inference_new_subjects import infer_batch_subjects

# Infer multiple subjects at once
results_df = infer_batch_subjects(
    model,
    X_new_batch,  # (n_subjects, 62)
    feature_names,
    dx_labels,
    subject_ids=['SUBJ_001', 'SUBJ_002', ...]
)

# Save results
results_df.to_csv('inference_results.csv', index=False)
```

## Model Architecture

### Mathematical Formulation

```
For each topic k = 1...K:
  β_k ~ Normal(0, σ)              [topic-region atrophy pattern]

For each patient d = 1...D:
  θ_d ~ Dirichlet(α)              [patient topic mixture]

  For each region v = 1...V:
    x_dv ~ Normal(θ_d @ β_v, σ_x) [observed atrophy]

  y_d ~ Categorical(Softmax(θ_d @ η)) [diagnosis]
```

### Key Differences from Text sLDA

| Aspect | Text sLDA | Our sLDA |
|--------|-----------|----------|
| Features | Word counts (discrete) | Atrophy values (continuous) |
| Likelihood | Multinomial | Normal |
| Response | Continuous (sentiment) | Categorical (diagnosis) |
| Supervised | Linear regression | Softmax classification |

## Interpretation Guide

### Topic Patterns (β)

- **K × 62 matrix** where each row is a topic
- Positive values: regions with high atrophy in this pattern
- Negative values: regions with low atrophy in this pattern
- Example topics:
  - Topic 0: Limbic/temporal (AD-like)
  - Topic 1: Cortical/parietal (DLB-like)
  - Topic 2: Subcortical sparing (PD-like)
  - Topic 3: Minimal atrophy (HC-like)

### Patient Mixtures (θ)

- **N × K matrix** where each row is a patient
- Values sum to 1 (proportions)
- **Single dominant topic**: Pure pathology
- **Mixed topics**: Co-pathology
- **High entropy**: Multiple overlapping patterns

### Diagnosis Weights (η)

- **K × 5 matrix** linking topics to diagnoses
- Positive: topic increases probability of diagnosis
- Negative: topic decreases probability of diagnosis
- Shows which pathology patterns predict which diagnoses

## Inferring New Subjects

### What Gets Inferred?

1. **Topic Mixture (θ_new)**:
   - Solved via least squares: `X_new ≈ θ_new @ β`
   - Represents the co-pathology pattern

2. **Diagnosis Probabilities**:
   - Computed via: `P(DX | θ_new) = Softmax(θ_new @ η)`
   - Shows likelihood of each diagnosis

3. **Predicted Diagnosis**:
   - `argmax(P(DX | θ_new))`

### Example Output

```
======================================================================
Inference Results for PATIENT_001
======================================================================

🎯 Predicted Diagnosis: AD
   Confidence: 67.3%

📊 Diagnosis Probability Breakdown:
Diagnosis       Probability  Bar
--------------------------------------------------
AD                  67.3%   ███████████████████████████
DLB                 18.2%   ███████
SVAD                 8.5%   ███
PD                   4.1%   █
HC                   1.9%

🧩 Co-Pathology Pattern (Topic Mixture):
Topic      Proportion   Bar
--------------------------------------------------
Topic 0       55.2%     ██████████████████████
Topic 1       28.3%     ███████████
Topic 2       12.1%     ████
Topic 3        4.4%     █

💡 Interpretation:
   ⚠ Multiple topics dominant → Co-pathology detected
   This subject shows mixed pathology patterns
```

### Interpreting Co-Pathology

| Pattern | Interpretation |
|---------|----------------|
| **Single topic (>80%)** | Pure pathology (e.g., typical AD) |
| **Two topics (40-60% each)** | Clear co-pathology (e.g., AD+DLB) |
| **Balanced (25% each)** | Complex/atypical presentation |
| **Low confidence (<50%)** | Uncertain diagnosis, needs clinical review |

## Visualization

```python
from visualization import *

# Topic-region heatmap
plot_topic_heatmap(topic_patterns, feature_names)

# Patient distributions by diagnosis
plot_patient_topic_distribution(patient_mixtures, y, dx_labels)

# Topic-diagnosis associations
plot_topic_diagnosis_association(diagnosis_weights, dx_labels)

# Detailed topic analysis
plot_brain_topic_pattern(0, topic_patterns, feature_names)

# Co-pathology examples
plot_copathology_mixtures(patient_mixtures, y, dx_labels)
```

## Requirements

```
pymc >= 5.0
pytensor
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
arviz
```

## Examples

See the notebooks:
- **Training**: `slda_copathology_example.ipynb`
- **Inference**: `inference_example.ipynb`

## Citation

If you use this implementation, please cite:

```
Supervised LDA for Co-Pathology Analysis in Neurodegenerative Diseases
Implementation based on:
- Blei & McAuliffe (2007). Supervised Topic Models. NIPS.
- Adapted for continuous neuroimaging features and categorical outcomes.
```

## Advanced Usage

### Custom Number of Topics

```python
# Try different numbers to find optimal
for n_topics in [3, 4, 5, 6]:
    model = CoPathologySLDA(n_topics=n_topics)
    model.fit(X, y)
    # Evaluate model fit...
```

### Prediction with Uncertainty

```python
# For full Bayesian prediction (not just point estimates)
# Use posterior predictive sampling in PyMC
# See model.trace_ for posterior samples
```

### Feature Selection

```python
# Use only specific regions
selected_features = ['ctx_lh_hippocampus', 'ctx_rh_hippocampus', ...]
X_selected = df[selected_features].values
```

## Troubleshooting

### Convergence Issues

```python
# If R-hat > 1.01, try:
model.fit(
    X, y,
    n_samples=3000,      # More samples
    tune=2000,           # More tuning
    target_accept=0.95   # Higher acceptance rate
)
```

### Low Prediction Accuracy

- Check if topics are interpretable
- Try different number of topics
- Ensure features are properly normalized
- Check for class imbalance

### Memory Issues

```python
# Use fewer chains or samples
model.fit(X, y, n_samples=1000, chains=2)

# Or use haiku model for faster inference
# (See PyMC documentation)
```

## Contact & Support

For questions, issues, or contributions, please refer to the implementation files and notebooks provided.
