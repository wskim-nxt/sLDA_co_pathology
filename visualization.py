"""
Visualization utilities for supervised LDA co-pathology analysis.

This module provides functions to visualize and interpret the learned
topic patterns, patient mixtures, and diagnosis associations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_topic_heatmap(
    topic_patterns: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
):
    """
    Plot heatmap of topic-region associations.

    Parameters
    ----------
    topic_patterns : np.ndarray, shape (n_topics, n_features)
        Topic-region weight matrix from model.get_topic_patterns()
    feature_names : List[str]
        Names of cortical features
    figsize : Tuple[int, int], default=(14, 6)
        Figure size
    cmap : str, default='RdBu_r'
        Colormap name
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics, n_features = topic_patterns.shape

    # Shorten feature names for readability (remove ctx_lh_ and ctx_rh_ prefixes)
    short_names = [name.replace('ctx_lh_', 'L_').replace('ctx_rh_', 'R_') for name in feature_names]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    vmax = np.abs(topic_patterns).max()
    sns.heatmap(
        topic_patterns,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=short_names,
        yticklabels=[f'Topic {i}' for i in range(n_topics)],
        cbar_kws={'label': 'Atrophy weight'},
        ax=ax
    )

    ax.set_xlabel('Brain Region', fontsize=12)
    ax.set_ylabel('Pathology Pattern (Topic)', fontsize=12)
    ax.set_title('Topic-Region Association Heatmap', fontsize=14, fontweight='bold')

    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_patient_topic_distribution(
    patient_mixtures: np.ndarray,
    diagnoses: np.ndarray,
    dx_labels: List[str],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot distribution of topic mixtures grouped by diagnosis.

    Parameters
    ----------
    patient_mixtures : np.ndarray, shape (n_patients, n_topics)
        Patient-topic proportion matrix from model.get_patient_mixtures()
    diagnoses : np.ndarray, shape (n_patients,)
        Integer-encoded diagnosis labels
    dx_labels : List[str]
        Diagnosis label names
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics = patient_mixtures.shape[1]
    n_classes = len(dx_labels)

    fig, axes = plt.subplots(1, n_topics, figsize=figsize, sharey=True)

    if n_topics == 1:
        axes = [axes]

    for topic_id in range(n_topics):
        ax = axes[topic_id]

        # Get topic proportions for this topic
        topic_props = patient_mixtures[:, topic_id]

        # Group by diagnosis
        data_by_dx = [topic_props[diagnoses == i] for i in range(n_classes)]

        # Box plot
        bp = ax.boxplot(
            data_by_dx,
            labels=dx_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
        )

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Diagnosis', fontsize=11)
        ax.set_ylabel('Topic Proportion', fontsize=11)
        ax.set_title(f'Topic {topic_id}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Patient Topic Mixtures by Diagnosis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_topic_diagnosis_association(
    diagnosis_weights: np.ndarray,
    dx_labels: List[str],
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'coolwarm',
    save_path: Optional[str] = None
):
    """
    Plot heatmap of topic-diagnosis associations.

    Parameters
    ----------
    diagnosis_weights : np.ndarray, shape (n_topics, n_classes)
        Topic-diagnosis weight matrix from model.get_diagnosis_weights()
    dx_labels : List[str]
        Diagnosis label names
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    cmap : str, default='coolwarm'
        Colormap name
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics, n_classes = diagnosis_weights.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    vmax = np.abs(diagnosis_weights).max()
    sns.heatmap(
        diagnosis_weights,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=dx_labels,
        yticklabels=[f'Topic {i}' for i in range(n_topics)],
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Association weight'},
        ax=ax
    )

    ax.set_xlabel('Diagnosis', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)
    ax.set_title('Topic-Diagnosis Association Matrix (η)', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dx_labels: List[str],
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for diagnosis predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True diagnosis labels
    y_pred : np.ndarray
        Predicted diagnosis labels
    dx_labels : List[str]
        Diagnosis label names
    figsize : Tuple[int, int], default=(8, 7)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=dx_labels,
        yticklabels=dx_labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Diagnosis', fontsize=12)
    ax.set_ylabel('True Diagnosis', fontsize=12)
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_brain_topic_pattern(
    topic_id: int,
    topic_patterns: np.ndarray,
    feature_names: List[str],
    n_top_regions: int = 15,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    Plot top regions for a specific topic, separated by hemisphere.

    Parameters
    ----------
    topic_id : int
        Topic index to visualize
    topic_patterns : np.ndarray, shape (n_topics, n_features)
        Topic-region weight matrix
    feature_names : List[str]
        Names of cortical features
    n_top_regions : int, default=15
        Number of top regions to show per hemisphere
    figsize : Tuple[int, int], default=(12, 5)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    topic_weights = topic_patterns[topic_id, :]

    # Separate left and right hemisphere
    lh_idx = [i for i, name in enumerate(feature_names) if 'ctx_lh_' in name]
    rh_idx = [i for i, name in enumerate(feature_names) if 'ctx_rh_' in name]

    lh_weights = topic_weights[lh_idx]
    rh_weights = topic_weights[rh_idx]

    lh_names = [feature_names[i].replace('ctx_lh_', '') for i in lh_idx]
    rh_names = [feature_names[i].replace('ctx_rh_', '') for i in rh_idx]

    # Get top regions by absolute value
    lh_top_idx = np.argsort(np.abs(lh_weights))[-n_top_regions:]
    rh_top_idx = np.argsort(np.abs(rh_weights))[-n_top_regions:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left hemisphere
    y_pos = np.arange(len(lh_top_idx))
    colors = ['red' if w > 0 else 'blue' for w in lh_weights[lh_top_idx]]

    ax1.barh(y_pos, lh_weights[lh_top_idx], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([lh_names[i] for i in lh_top_idx], fontsize=9)
    ax1.set_xlabel('Atrophy Weight', fontsize=11)
    ax1.set_title(f'Left Hemisphere - Topic {topic_id}', fontsize=12, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)

    # Right hemisphere
    y_pos = np.arange(len(rh_top_idx))
    colors = ['red' if w > 0 else 'blue' for w in rh_weights[rh_top_idx]]

    ax2.barh(y_pos, rh_weights[rh_top_idx], color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([rh_names[i] for i in rh_top_idx], fontsize=9)
    ax2.set_xlabel('Atrophy Weight', fontsize=11)
    ax2.set_title(f'Right Hemisphere - Topic {topic_id}', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle(
        f'Top Regional Associations for Topic {topic_id}\n(Red=positive, Blue=negative)',
        fontsize=13,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_copathology_mixtures(
    patient_mixtures: np.ndarray,
    diagnoses: np.ndarray,
    dx_labels: List[str],
    patient_ids: Optional[List[str]] = None,
    n_patients_per_dx: int = 5,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot stacked bar charts showing co-pathology (topic mixtures) for example patients.

    Parameters
    ----------
    patient_mixtures : np.ndarray, shape (n_patients, n_topics)
        Patient-topic proportion matrix
    diagnoses : np.ndarray, shape (n_patients,)
        Integer-encoded diagnosis labels
    dx_labels : List[str]
        Diagnosis label names
    patient_ids : List[str], optional
        Patient identifiers for labels
    n_patients_per_dx : int, default=5
        Number of example patients to show per diagnosis
    figsize : Tuple[int, int], default=(14, 8)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics = patient_mixtures.shape[1]
    n_classes = len(dx_labels)

    # Select example patients from each diagnosis
    selected_patients = []
    selected_labels = []
    selected_mixtures = []

    for dx_id in range(n_classes):
        dx_patients = np.where(diagnoses == dx_id)[0]
        if len(dx_patients) == 0:
            continue

        # Select up to n_patients_per_dx
        n_select = min(n_patients_per_dx, len(dx_patients))
        selected_idx = np.random.choice(dx_patients, size=n_select, replace=False)

        for idx in selected_idx:
            selected_patients.append(idx)
            if patient_ids:
                selected_labels.append(f"{dx_labels[dx_id]}_{patient_ids[idx]}")
            else:
                selected_labels.append(f"{dx_labels[dx_id]}_{idx}")
            selected_mixtures.append(patient_mixtures[idx, :])

    selected_mixtures = np.array(selected_mixtures)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(selected_patients))
    colors = plt.cm.Set2(np.linspace(0, 1, n_topics))

    bottom = np.zeros(len(selected_patients))
    for topic_id in range(n_topics):
        ax.bar(
            x,
            selected_mixtures[:, topic_id],
            bottom=bottom,
            label=f'Topic {topic_id}',
            color=colors[topic_id],
            alpha=0.8
        )
        bottom += selected_mixtures[:, topic_id]

    ax.set_xticks(x)
    ax.set_xticklabels(selected_labels, rotation=90, ha='right', fontsize=8)
    ax.set_ylabel('Topic Proportion', fontsize=12)
    ax.set_xlabel('Patient', fontsize=12)
    ax.set_title('Co-Pathology Patterns: Topic Mixtures per Patient', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_topic_composition_radar(
    topic_patterns: np.ndarray,
    feature_names: List[str],
    n_top_regions: int = 15,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot radar/spider charts showing topic composition by brain regions.

    This provides an intuitive view of which regions define each topic.

    Parameters
    ----------
    topic_patterns : np.ndarray, shape (n_topics, n_features)
        Topic-region weight matrix
    feature_names : List[str]
        Names of cortical features
    n_top_regions : int, default=15
        Number of top regions to show per topic
    figsize : Tuple[int, int], default=(14, 10)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics = topic_patterns.shape[0]

    # Create subplot grid
    n_cols = 2
    n_rows = (n_topics + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             subplot_kw=dict(projection='polar'))

    if n_topics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for topic_id in range(n_topics):
        ax = axes[topic_id]

        # Get top regions by absolute value
        topic_weights = topic_patterns[topic_id, :]
        top_idx = np.argsort(np.abs(topic_weights))[-n_top_regions:][::-1]

        top_weights = topic_weights[top_idx]
        top_names = [feature_names[i].replace('ctx_lh_', 'L_').replace('ctx_rh_', 'R_')
                     for i in top_idx]

        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(top_weights), endpoint=False).tolist()
        top_weights_plot = top_weights.tolist()

        # Close the plot
        angles += angles[:1]
        top_weights_plot += top_weights_plot[:1]

        # Plot
        ax.plot(angles, top_weights_plot, 'o-', linewidth=2, label=f'Topic {topic_id}')
        ax.fill(angles, top_weights_plot, alpha=0.25)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_names, size=8)
        ax.set_ylim(min(0, min(top_weights_plot)) * 1.2, max(top_weights_plot) * 1.2)
        ax.set_title(f'Topic {topic_id} Composition', size=12, fontweight='bold', pad=20)
        ax.grid(True)

    # Hide unused subplots
    for idx in range(n_topics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Topic Composition: Top Regional Patterns',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_topic_wordcloud_style(
    topic_patterns: np.ndarray,
    feature_names: List[str],
    n_top_regions: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot topic composition as word-cloud style bar charts.

    Shows the most important regions for each topic with bar size
    proportional to weight magnitude.

    Parameters
    ----------
    topic_patterns : np.ndarray, shape (n_topics, n_features)
        Topic-region weight matrix
    feature_names : List[str]
        Names of cortical features
    n_top_regions : int, default=20
        Number of top regions to show per topic
    figsize : Tuple[int, int], default=(14, 8)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics = topic_patterns.shape[0]

    fig, axes = plt.subplots(n_topics, 1, figsize=figsize)

    if n_topics == 1:
        axes = [axes]

    for topic_id in range(n_topics):
        ax = axes[topic_id]

        # Get top regions by absolute value
        topic_weights = topic_patterns[topic_id, :]
        top_idx = np.argsort(np.abs(topic_weights))[-n_top_regions:]

        top_weights = topic_weights[top_idx]
        top_names = [feature_names[i].replace('ctx_lh_', 'L_').replace('ctx_rh_', 'R_')
                     for i in top_idx]

        # Color by positive/negative
        colors = ['#e74c3c' if w > 0 else '#3498db' for w in top_weights]

        # Horizontal bar chart
        y_pos = np.arange(len(top_weights))
        ax.barh(y_pos, top_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('Atrophy Weight', fontsize=10)
        ax.set_title(f'Topic {topic_id}: Top {n_top_regions} Regions',
                    fontsize=11, fontweight='bold', pad=10)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (weight, name) in enumerate(zip(top_weights, top_names)):
            x_pos = weight + (0.02 if weight > 0 else -0.02)
            ha = 'left' if weight > 0 else 'right'
            ax.text(x_pos, i, f'{weight:.3f}', va='center', ha=ha, fontsize=7)

    plt.suptitle('Topic Composition: Regional Atrophy Weights\n(Red=Positive, Blue=Negative)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_topic_comparison(
    topic_patterns: np.ndarray,
    feature_names: List[str],
    topic_ids: List[int] = None,
    n_top_regions: int = 15,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple topics side-by-side to see their differences.

    Parameters
    ----------
    topic_patterns : np.ndarray, shape (n_topics, n_features)
        Topic-region weight matrix
    feature_names : List[str]
        Names of cortical features
    topic_ids : List[int], optional
        Which topics to compare (default: all)
    n_top_regions : int, default=15
        Number of regions to show
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_topics = topic_patterns.shape[0]

    if topic_ids is None:
        topic_ids = list(range(n_topics))

    # Find union of top regions across selected topics
    all_top_regions = set()
    for topic_id in topic_ids:
        topic_weights = np.abs(topic_patterns[topic_id, :])
        top_idx = np.argsort(topic_weights)[-n_top_regions:]
        all_top_regions.update(top_idx)

    # Sort by average importance across topics
    region_indices = list(all_top_regions)
    avg_weights = np.mean(np.abs(topic_patterns[:, region_indices]), axis=0)
    sorted_idx = [region_indices[i] for i in np.argsort(avg_weights)[::-1]]

    # Prepare data for grouped bar chart
    region_names = [feature_names[i].replace('ctx_lh_', 'L_').replace('ctx_rh_', 'R_')
                    for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(region_names))
    width = 0.8 / len(topic_ids)

    colors = plt.cm.Set2(np.linspace(0, 1, len(topic_ids)))

    for i, topic_id in enumerate(topic_ids):
        weights = topic_patterns[topic_id, sorted_idx]
        offset = width * (i - len(topic_ids) / 2 + 0.5)
        ax.bar(x + offset, weights, width, label=f'Topic {topic_id}',
               color=colors[i], alpha=0.8)

    ax.set_xlabel('Brain Region', fontsize=11)
    ax.set_ylabel('Atrophy Weight', fontsize=11)
    ax.set_title('Topic Comparison: Regional Patterns', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=1)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_topic_summary(
    topic_patterns: np.ndarray,
    feature_names: List[str],
    diagnosis_weights: np.ndarray,
    dx_labels: List[str],
    patient_mixtures: Optional[np.ndarray] = None,
    diagnoses: Optional[np.ndarray] = None,
    topic_id: int = 0,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Comprehensive summary of a single topic showing:
    - Top regions (bar chart)
    - Hemisphere split (left vs right)
    - Association with diagnoses
    - (Optional) Distribution in patient population

    Parameters
    ----------
    topic_patterns : np.ndarray
        Topic-region weight matrix
    feature_names : List[str]
        Names of cortical features
    diagnosis_weights : np.ndarray
        Topic-diagnosis association matrix
    dx_labels : List[str]
        Diagnosis labels
    patient_mixtures : np.ndarray, optional
        Patient-topic proportions
    diagnoses : np.ndarray, optional
        Patient diagnoses (for distribution plot)
    topic_id : int, default=0
        Which topic to summarize
    figsize : Tuple[int, int], default=(16, 10)
        Figure size
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    topic_weights = topic_patterns[topic_id, :]

    # Split by hemisphere
    lh_idx = [i for i, name in enumerate(feature_names) if 'ctx_lh_' in name]
    rh_idx = [i for i, name in enumerate(feature_names) if 'ctx_rh_' in name]

    lh_weights = topic_weights[lh_idx]
    rh_weights = topic_weights[rh_idx]
    lh_names = [feature_names[i].replace('ctx_lh_', '') for i in lh_idx]
    rh_names = [feature_names[i].replace('ctx_rh_', '') for i in rh_idx]

    # Create figure with subplots
    if patient_mixtures is not None and diagnoses is not None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Top regions
        ax2 = fig.add_subplot(gs[1, 0])  # Left hemisphere
        ax3 = fig.add_subplot(gs[1, 1])  # Right hemisphere
        ax4 = fig.add_subplot(gs[2, 0])  # Diagnosis associations
        ax5 = fig.add_subplot(gs[2, 1])  # Patient distribution
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.7))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Top regions
        ax2 = fig.add_subplot(gs[1, 0])  # Left hemisphere
        ax3 = fig.add_subplot(gs[1, 1])  # Right hemisphere
        ax4 = None
        ax5 = None

    # 1. Top regions overall
    top_n = 15
    top_idx = np.argsort(np.abs(topic_weights))[-top_n:]
    top_weights = topic_weights[top_idx]
    top_names = [feature_names[i].replace('ctx_lh_', 'L_').replace('ctx_rh_', 'R_')
                 for i in top_idx]
    colors = ['#e74c3c' if w > 0 else '#3498db' for w in top_weights]

    y_pos = np.arange(len(top_weights))
    ax1.barh(y_pos, top_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_names, fontsize=9)
    ax1.set_xlabel('Atrophy Weight', fontsize=10)
    ax1.set_title(f'Topic {topic_id}: Top {top_n} Regions (Red=High, Blue=Low)',
                  fontsize=12, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=1.5)
    ax1.grid(axis='x', alpha=0.3)

    # 2. Left hemisphere
    lh_top_n = 10
    lh_top_idx = np.argsort(np.abs(lh_weights))[-lh_top_n:]
    colors_lh = ['#e74c3c' if w > 0 else '#3498db' for w in lh_weights[lh_top_idx]]

    y_pos = np.arange(len(lh_top_idx))
    ax2.barh(y_pos, lh_weights[lh_top_idx], color=colors_lh, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lh_names[i] for i in lh_top_idx], fontsize=8)
    ax2.set_xlabel('Weight', fontsize=9)
    ax2.set_title('Left Hemisphere', fontsize=11, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    # 3. Right hemisphere
    rh_top_n = 10
    rh_top_idx = np.argsort(np.abs(rh_weights))[-rh_top_n:]
    colors_rh = ['#e74c3c' if w > 0 else '#3498db' for w in rh_weights[rh_top_idx]]

    y_pos = np.arange(len(rh_top_idx))
    ax3.barh(y_pos, rh_weights[rh_top_idx], color=colors_rh, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([rh_names[i] for i in rh_top_idx], fontsize=8)
    ax3.set_xlabel('Weight', fontsize=9)
    ax3.set_title('Right Hemisphere', fontsize=11, fontweight='bold')
    ax3.axvline(0, color='black', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)

    # 4. Diagnosis associations (if we have the bottom row)
    if ax4 is not None:
        dx_weights = diagnosis_weights[topic_id, :]
        colors_dx = ['green' if w > 0 else 'red' for w in dx_weights]

        ax4.bar(dx_labels, dx_weights, color=colors_dx, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Association Weight', fontsize=10)
        ax4.set_title('Diagnosis Associations', fontsize=11, fontweight='bold')
        ax4.axhline(0, color='black', linewidth=1)
        ax4.grid(axis='y', alpha=0.3)

    # 5. Patient distribution (if data provided)
    if ax5 is not None and patient_mixtures is not None:
        topic_props = patient_mixtures[:, topic_id]
        n_classes = len(dx_labels)
        data_by_dx = [topic_props[diagnoses == i] for i in range(n_classes)]

        bp = ax5.boxplot(data_by_dx, labels=dx_labels, patch_artist=True, showmeans=True)
        colors_box = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)

        ax5.set_ylabel('Topic Proportion', fontsize=10)
        ax5.set_title('Distribution Across Diagnoses', fontsize=11, fontweight='bold')
        ax5.set_ylim(0, 1)
        ax5.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Topic {topic_id}: Comprehensive Summary',
                 fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Test visualizations with synthetic data
    print("Testing visualization functions with synthetic data...\n")

    np.random.seed(42)

    # Generate synthetic data
    n_patients = 100
    n_features = 20
    n_topics = 3
    n_classes = 3

    topic_patterns = np.random.randn(n_topics, n_features)
    patient_mixtures = np.random.dirichlet([1] * n_topics, size=n_patients)
    diagnoses = np.random.choice(n_classes, size=n_patients)
    diagnosis_weights = np.random.randn(n_topics, n_classes)

    feature_names = [f"region_{i}" for i in range(n_features)]
    dx_labels = ['AD', 'PD', 'HC']

    # Test each plot
    print("Creating topic heatmap...")
    plot_topic_heatmap(topic_patterns, feature_names, save_path='/home/coder/sLDA_co_pathology/figures/topic_heatmap.png')

    print("Creating patient topic distribution...")
    plot_patient_topic_distribution(patient_mixtures, diagnoses, dx_labels, save_path='/home/coder/sLDA_co_pathology/figures/patient_topic_distribution.png')

    print("Creating topic-diagnosis association...")
    plot_topic_diagnosis_association(diagnosis_weights, dx_labels, save_path='/home/coder/sLDA_co_pathology/figures/topic_diagnosis_association.png')

    print("Creating brain topic pattern...")
    plot_brain_topic_pattern(0, topic_patterns, feature_names, save_path='/home/coder/sLDA_co_pathology/figures/topic_pattern.png')

    print("Creating co-pathology mixtures...")
    plot_copathology_mixtures(patient_mixtures, diagnoses, dx_labels, n_patients_per_dx=3, save_path='/home/coder/sLDA_co_pathology/figures/copathology_mixtures.png')

    print("\n✓ All visualizations created successfully!")
    plt.show()
