import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from preprocessing import load_wsev_data, prepare_slda_inputs, train_test_split_stratified
from slda_model import CoPathologySLDA
from visualization import (
    plot_topic_heatmap,
    plot_patient_topic_distribution,
    plot_topic_diagnosis_association,
    plot_brain_topic_pattern,
    plot_copathology_mixtures,
    plot_confusion_matrix
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Load the WSEV dataset
data_path = '/home/coder/data/updated_WSEV/260108_wsev_final_df.csv'
df = load_wsev_data(data_path)

# Display basic info
print(f"\nDataset shape: {df.shape}")
print(f"\nDiagnosis distribution:")
print(df['DX'].value_counts())

# Prepare data for sLDA
X, y, feature_names, dx_labels = prepare_slda_inputs(df, standardize=False)

print(X)
print(feature_names)
print(f"\nFeature matrix X: {X.shape}")
print(f"Diagnosis labels y: {y.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Diagnoses: {dx_labels}")

# Use full dataset
X_model, y_model = X, y

# Initialize model
model = CoPathologySLDA(
    n_topics=6,           # Number of latent pathology patterns
    alpha_prior=1.0,      # Dirichlet concentration (1.0 = uniform)
    feature_prior_std=1.0, # Prior std for topic-region weights
    random_state=42
)

print("Model initialized with 4 topics")
print("\nUsing ADVI - this should take ~30-60 seconds...")

# Fit the model using ADVI
model.fit(
    X_model, 
    y_model,
    inference='advi',        # Use variational inference
    n_advi_iterations=10000, # Optimization iterations
    n_samples=1000           # Samples from approximate posterior
)

# Plot ELBO convergence
fig = model.plot_elbo(figsize=(12, 4), save_path='/home/coder/sLDA_co_pathology/figures/ADVI/advi_elbo.png')
plt.show()
# Get posterior means
topic_patterns = model.get_topic_patterns()        # (n_topics, n_features)
patient_mixtures = model.get_patient_mixtures()    # (n_patients, n_topics)
diagnosis_weights = model.get_diagnosis_weights()  # (n_topics, n_classes)

print(f"Topic patterns (beta): {topic_patterns.shape}")
print(f"Patient mixtures (theta): {patient_mixtures.shape}")
print(f"Diagnosis weights (eta): {diagnosis_weights.shape}")

# Verify topic mixtures sum to 1
print(f"\nPatient mixture sums (should be ~1.0): {patient_mixtures[0].sum():.4f}")


fig = plot_topic_heatmap(
    topic_patterns, 
    feature_names,
    figsize=(16, 6),
    save_path='/home/coder/sLDA_co_pathology/figures/ADVI/topic_heatmap.png'
)
plt.show()

# Print top regions for each topic
for topic_id in range(model.n_topics):
    print(f"\n{'='*60}")
    print(f"Topic {topic_id} - Top 10 Regions")
    print(f"{'='*60}")
    
    top_regions = model.get_topic_top_regions(
        topic_id, 
        feature_names, 
        n_regions=10,
        absolute=True
    )
    
    for i, (region, weight) in enumerate(top_regions, 1):
        print(f"{i:2d}. {region:40s} {weight:+.3f}")

# Visualize each topic
for topic_id in range(model.n_topics):
    fig = plot_brain_topic_pattern(
        topic_id,
        topic_patterns,
        feature_names,
        n_top_regions=12,
        figsize=(14, 5),
        save_path=f'/home/coder/sLDA_co_pathology/figures/ADVI/topic_{topic_id}_brain_pattern.png'
    )
    plt.show()

fig = plot_patient_topic_distribution(
    patient_mixtures,
    y_model,
    dx_labels,
    figsize=(14, 6),
    save_path='/home/coder/sLDA_co_pathology/figures/ADVI/patient_topic_distribution.png'
)
plt.show()

fig = plot_topic_diagnosis_association(
    diagnosis_weights,
    dx_labels,
    figsize=(9, 6),
    save_path='/home/coder/sLDA_co_pathology/figures/ADVI/topic_diagnosis_association.png'
)
plt.show()
# Interpret topic-diagnosis associations
print("Topic-Diagnosis Associations (η matrix):\n")
print(f"{'Topic':<10}", end="")
for dx in dx_labels:
    print(f"{dx:>10}", end="")
print("\n" + "="*60)

for topic_id in range(model.n_topics):
    print(f"Topic {topic_id:<4}", end="")
    for dx_id in range(len(dx_labels)):
        weight = diagnosis_weights[topic_id, dx_id]
        print(f"{weight:>10.3f}", end="")
    print()

print("\nInterpretation:")
print("- Positive weights: Topic increases probability of diagnosis")
print("- Negative weights: Topic decreases probability of diagnosis")


# Save topic patterns
topic_df = pd.DataFrame(
    topic_patterns.T,
    index=feature_names,
    columns=[f'Topic_{i}' for i in range(model.n_topics)]
)
topic_df.to_csv('/home/coder/sLDA_co_pathology/results/ADVI/260112_topic_patterns.csv')
print("Saved topic patterns to topic_patterns.csv")

# Save patient mixtures
mixture_df = pd.DataFrame(
    patient_mixtures,
    columns=[f'Topic_{i}' for i in range(model.n_topics)]
)
mixture_df['Diagnosis'] = [dx_labels[i] for i in y_model]

# Save diagnosis weights
dx_weights_df = pd.DataFrame(
    diagnosis_weights,
    index=[f'Topic_{i}' for i in range(model.n_topics)],
    columns=dx_labels
)
dx_weights_df.to_csv('/home/coder/sLDA_co_pathology/results/ADVI/260112_topic_diagnosis_weights.csv')
print("Saved diagnosis weights to topic_diagnosis_weights.csv")

# -------------------------------
# Create Diagnosis x Topic CSV (average topic expression per diagnosis)
# -------------------------------

# Create a DataFrame with patient topic mixtures and diagnosis
patient_topic_df = pd.DataFrame(
    patient_mixtures,
    columns=[f'Topic_{i}' for i in range(model.n_topics)]
)
patient_topic_df['Diagnosis'] = [dx_labels[i] for i in y_model]

# Group by Diagnosis and compute mean topic mixture
dx_topic_df = patient_topic_df.groupby('Diagnosis').mean()

# Optional: convert to percentages (each row sums to 100%)
dx_topic_percent_df = dx_topic_df.div(dx_topic_df.sum(axis=1), axis=0) * 100

# Save to CSV
dx_topic_csv_path = '/home/coder/sLDA_co_pathology/results/ADVI/260112_dx_topic_percent.csv'
dx_topic_percent_df.to_csv(dx_topic_csv_path)
print(f"Saved Diagnosis x Topic CSV to {dx_topic_csv_path}")