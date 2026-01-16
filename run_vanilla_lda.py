"""
Run Vanilla LDA analysis on WSEV data to discover atrophy patterns (topics).

This script:
1. Loads and prepares the data
2. Fits an unsupervised LDA model
3. Extracts topic patterns (atrophy signatures)
4. Visualizes topic distributions across diagnosis groups with a spider plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from vanilla_lda_model import VanillaLDA
from preprocessing import load_wsev_data, prepare_slda_inputs


def plot_topic_spider(
    theta: np.ndarray,
    y: np.ndarray,
    dx_labels: list,
    title: str = "Topic Distribution by Diagnosis Group",
    figsize: tuple = (10, 10),
    save_path: str = None,
    colors: list = None
):
    """
    Create a spider/radar plot showing topic distributions for each diagnosis group.

    Parameters
    ----------
    theta : np.ndarray, shape (n_patients, n_topics)
        Patient topic mixtures
    y : np.ndarray, shape (n_patients,)
        Diagnosis labels (encoded as integers)
    dx_labels : list
        Names of diagnosis groups
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    colors : list, optional
        Colors for each diagnosis group

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_topics = theta.shape[1]
    n_groups = len(dx_labels)

    # Compute mean topic proportions for each diagnosis group
    group_means = np.zeros((n_groups, n_topics))
    group_stds = np.zeros((n_groups, n_topics))

    for i, dx in enumerate(dx_labels):
        mask = y == i
        group_means[i] = theta[mask].mean(axis=0)
        group_stds[i] = theta[mask].std(axis=0)

    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, n_topics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Default colors
    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

    # Plot each diagnosis group
    for i, (dx, color) in enumerate(zip(dx_labels, colors)):
        values = group_means[i].tolist()
        values += values[:1]  # Complete the loop

        stds = group_stds[i].tolist()
        stds += stds[:1]

        # Plot the line
        ax.plot(angles, values, 'o-', linewidth=2, label=dx, color=color)

        # Fill with transparency
        ax.fill(angles, values, alpha=0.15, color=color)

        # Add error bands (optional - can be visually noisy)
        values_upper = [v + s for v, s in zip(values, stds)]
        values_lower = [max(0, v - s) for v, s in zip(values, stds)]
        ax.fill_between(angles, values_lower, values_upper, alpha=0.05, color=color)

    # Customize the plot
    topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(topic_labels, size=12, fontweight='bold')

    # Set radial limits
    ax.set_ylim(0, max(group_means.max() * 1.2, 0.5))

    # Add gridlines
    ax.set_rgrids([0.1, 0.2, 0.3, 0.4, 0.5], angle=0, fontsize=9)

    ax.set_title(title, size=16, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spider plot saved to: {save_path}")

    return fig


def plot_topic_heatmap(
    beta: np.ndarray,
    feature_names: list,
    n_top_features: int = 15,
    figsize: tuple = (14, 10),
    save_path: str = None
):
    """
    Create a heatmap showing top features for each topic.

    Parameters
    ----------
    beta : np.ndarray, shape (n_topics, n_features)
        Topic patterns
    feature_names : list
        Feature names
    n_top_features : int
        Number of top features to show per topic
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n_topics = beta.shape[0]

    fig, axes = plt.subplots(1, n_topics, figsize=figsize)
    if n_topics == 1:
        axes = [axes]

    for k in range(n_topics):
        ax = axes[k]

        # Get top features by absolute value
        topic_weights = beta[k]
        sorted_idx = np.argsort(np.abs(topic_weights))[::-1][:n_top_features]

        top_features = [feature_names[i] for i in sorted_idx]
        top_weights = topic_weights[sorted_idx]

        # Create horizontal bar chart
        colors = ['#d73027' if w < 0 else '#1a9850' for w in top_weights]
        y_pos = np.arange(len(top_features))

        ax.barh(y_pos, top_weights, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('ctx_', '').replace('_', ' ') for f in top_features], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(f'Topic {k+1}', fontweight='bold', fontsize=12)
        ax.axvline(x=0, color='black', linewidth=0.5)

    plt.suptitle('Top Brain Regions per Topic', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    return fig


def print_topic_summary(model: VanillaLDA, feature_names: list, n_regions: int = 10):
    """Print a summary of each topic's top regions."""
    print("\n" + "=" * 70)
    print("TOPIC SUMMARY - Top Brain Regions per Topic")
    print("=" * 70)

    for k in range(model.n_topics):
        print(f"\n--- Topic {k+1} ---")
        top_regions = model.get_topic_top_regions(k, feature_names, n_regions=n_regions)

        for region, weight in top_regions:
            clean_name = region.replace('ctx_', '').replace('_', ' ')
            direction = "↑" if weight > 0 else "↓"
            print(f"  {direction} {clean_name}: {weight:.3f}")


def main():
    # Configuration
    N_TOPICS = 8 #####
    INFERENCE = "advi"  # Use "mcmc" for more accurate (but slower) inference
    N_ADVI_ITERATIONS = 30000
    N_SAMPLES = 1000

    # Create output directory
    output_dir = Path("figures/vanilla_lda")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("=" * 70)
    print("VANILLA LDA ANALYSIS - Discovering Atrophy Patterns")
    print("=" * 70)

    data_path = '/home/coder/data/updated_WSEV/260108_wsev_final_df.csv'
    df = load_wsev_data(data_path)

    X, y, feature_names, dx_labels = prepare_slda_inputs(df, standardize=False)

    print(f"\nDiagnosis groups: {dx_labels}")
    print(f"Samples per group: {[np.sum(y == i) for i in range(len(dx_labels))]}")

    # Fit the model
    print("\n" + "=" * 70)
    print(f"FITTING VANILLA LDA MODEL ({N_TOPICS} topics)")
    print("=" * 70)

    model = VanillaLDA(
        n_topics=N_TOPICS,
        alpha_prior=1.0,
        feature_prior_std=1.0,
        random_state=42
    )

    model.fit(
        X,
        inference=INFERENCE,
        n_advi_iterations=N_ADVI_ITERATIONS,
        n_samples=N_SAMPLES
    )

    # Plot ELBO convergence (for ADVI)
    if INFERENCE == "advi":
        fig_elbo = model.plot_elbo(save_path=output_dir / "elbo_convergence.png")
        plt.close(fig_elbo)

    # Extract results
    beta = model.get_topic_patterns()   # (n_topics, n_features)
    theta = model.get_patient_mixtures()  # (n_patients, n_topics)

    print(f"\nExtracted topic patterns: {beta.shape}")
    print(f"Extracted patient mixtures: {theta.shape}")

    # Print topic summary
    print_topic_summary(model, feature_names, n_regions=10)

    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Spider plot: Topic distributions by diagnosis group
    fig_spider = plot_topic_spider(
        theta=theta,
        y=y,
        dx_labels=dx_labels,
        title="Topic Distribution by Diagnosis Group",
        save_path=output_dir / "topic_spider_plot.png"
    )
    plt.show()

    # 2. Topic heatmap: Top regions per topic
    fig_heatmap = plot_topic_heatmap(
        beta=beta,
        feature_names=feature_names,
        n_top_features=15,
        save_path=output_dir / "topic_heatmap.png"
    )
    plt.show()

    # 3. Additional: Box plot of topic distributions
    fig_box, axes = plt.subplots(1, N_TOPICS, figsize=(4 * N_TOPICS, 5))
    if N_TOPICS == 1:
        axes = [axes]

    for k in range(N_TOPICS):
        ax = axes[k]
        data_by_group = [theta[y == i, k] for i in range(len(dx_labels))]
        bp = ax.boxplot(data_by_group, labels=dx_labels, patch_artist=True)

        colors = plt.cm.Set2(np.linspace(0, 1, len(dx_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Topic Proportion')
        ax.set_title(f'Topic {k+1}', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Topic Proportions by Diagnosis Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_box.savefig(output_dir / "topic_boxplots.png", dpi=150, bbox_inches='tight')
    print(f"Box plots saved to: {output_dir / 'topic_boxplots.png'}")
    plt.show()

    # Print group statistics
    print("\n" + "=" * 70)
    print("GROUP STATISTICS - Mean Topic Proportions")
    print("=" * 70)

    stats_df = pd.DataFrame(index=dx_labels, columns=[f"Topic {k+1}" for k in range(N_TOPICS)])

    for i, dx in enumerate(dx_labels):
        mask = y == i
        mean_props = theta[mask].mean(axis=0)
        for k in range(N_TOPICS):
            stats_df.loc[dx, f"Topic {k+1}"] = f"{mean_props[k]:.3f}"

    print(stats_df.to_string())

    # Save statistics
    stats_df.to_csv(output_dir / "group_topic_statistics.csv")
    print(f"\nStatistics saved to: {output_dir / 'group_topic_statistics.csv'}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)

    return model, theta, beta, y, dx_labels, feature_names


if __name__ == "__main__":
    model, theta, beta, y, dx_labels, feature_names = main()
