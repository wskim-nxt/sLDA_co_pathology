"""
Inference utilities for predicting diagnosis and topic mixtures for new subjects.

This module provides functions to apply a trained sLDA model to new patients
and infer their co-pathology patterns and diagnosis probabilities.
"""

import numpy as np
from typing import Tuple, Dict, List
import pandas as pd


def infer_new_subject(
    model,
    X_new: np.ndarray,
    feature_names: List[str],
    dx_labels: List[str],
    subject_id: str = "New Subject"
) -> Dict:
    """
    Infer topic mixture and diagnosis probabilities for a new subject.

    Parameters
    ----------
    model : CoPathologySLDA
        Trained sLDA model
    X_new : np.ndarray, shape (n_features,) or (1, n_features)
        Regional atrophy values for the new subject
    feature_names : List[str]
        Feature names (must match training data)
    dx_labels : List[str]
        Diagnosis labels (must match training data)
    subject_id : str, default="New Subject"
        Identifier for the subject

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'subject_id': Subject identifier
        - 'topic_mixture': Array of topic proportions (sums to 1)
        - 'diagnosis_probs': Array of diagnosis probabilities
        - 'predicted_diagnosis': Most likely diagnosis label
        - 'top_topics': List of (topic_id, proportion) sorted by importance
        - 'diagnosis_breakdown': Dict of {diagnosis: probability}
    """
    # Ensure X_new is 2D
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    # Get diagnosis probabilities (this internally infers topic mixture)
    dx_probs = model.predict_diagnosis_proba(X_new)[0]

    # Infer topic mixture
    topic_mixture = infer_topic_mixture(model, X_new)[0]

    # Get predicted diagnosis
    pred_dx_idx = np.argmax(dx_probs)
    pred_dx = dx_labels[pred_dx_idx]

    # Sort topics by importance
    top_topics = sorted(
        [(i, topic_mixture[i]) for i in range(len(topic_mixture))],
        key=lambda x: x[1],
        reverse=True
    )

    # Diagnosis breakdown
    dx_breakdown = {dx_labels[i]: dx_probs[i] for i in range(len(dx_labels))}

    results = {
        'subject_id': subject_id,
        'topic_mixture': topic_mixture,
        'diagnosis_probs': dx_probs,
        'predicted_diagnosis': pred_dx,
        'top_topics': top_topics,
        'diagnosis_breakdown': dx_breakdown
    }

    return results


def infer_topic_mixture(
    model,
    X_new: np.ndarray,
    method: str = 'lstsq'
) -> np.ndarray:
    """
    Infer topic mixtures for new subjects given their atrophy patterns.

    This function solves for θ_new given: X_new ≈ θ_new @ β
    where β is the learned topic-region pattern matrix.

    Parameters
    ----------
    model : CoPathologySLDA
        Trained sLDA model
    X_new : np.ndarray, shape (n_new_patients, n_features)
        Regional atrophy values for new subjects
    method : str, default='lstsq'
        Method to use: 'lstsq' (least squares) or 'nnls' (non-negative least squares)

    Returns
    -------
    theta_new : np.ndarray, shape (n_new_patients, n_topics)
        Inferred topic proportions for each new subject
        Each row sums to 1
    """
    beta = model.get_topic_patterns()  # (K, V)

    if method == 'lstsq':
        from scipy.linalg import lstsq

        # Solve: X_new.T ≈ beta.T @ theta_new.T
        # So: theta_new.T ≈ (beta.T)^+ @ X_new.T
        theta_new, _, _, _ = lstsq(beta.T, X_new.T)
        theta_new = theta_new.T  # (N_new, K)

    elif method == 'nnls':
        from scipy.optimize import nnls

        # Non-negative least squares (enforces non-negativity)
        n_new = X_new.shape[0]
        n_topics = beta.shape[0]
        theta_new = np.zeros((n_new, n_topics))

        for i in range(n_new):
            theta_new[i], _ = nnls(beta.T, X_new[i])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lstsq' or 'nnls'")

    # Ensure non-negative and normalize to sum to 1
    theta_new = np.maximum(theta_new, 0)
    theta_new = theta_new / theta_new.sum(axis=1, keepdims=True)

    return theta_new


def print_inference_results(results: Dict, verbose: bool = True):
    """
    Pretty print inference results for a new subject.

    Parameters
    ----------
    results : dict
        Output from infer_new_subject()
    verbose : bool, default=True
        If True, print detailed topic mixture and diagnosis breakdown
    """
    print(f"\n{'='*70}")
    print(f"Inference Results for {results['subject_id']}")
    print(f"{'='*70}\n")

    # Predicted diagnosis
    print(f"🎯 Predicted Diagnosis: {results['predicted_diagnosis']}")
    print(f"   Confidence: {max(results['diagnosis_probs']):.1%}\n")

    # Diagnosis probabilities
    print("📊 Diagnosis Probability Breakdown:")
    print(f"{'Diagnosis':<15} {'Probability':<12} {'Bar'}")
    print("-" * 50)
    for dx, prob in sorted(results['diagnosis_breakdown'].items(),
                           key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"{dx:<15} {prob:>6.1%}       {bar}")

    if verbose:
        print(f"\n🧩 Co-Pathology Pattern (Topic Mixture):")
        print(f"{'Topic':<10} {'Proportion':<12} {'Bar'}")
        print("-" * 50)
        for topic_id, proportion in results['top_topics']:
            bar = "█" * int(proportion * 40)
            print(f"Topic {topic_id:<4} {proportion:>6.1%}       {bar}")

        print(f"\n💡 Interpretation:")
        if len([p for _, p in results['top_topics'] if p > 0.3]) > 1:
            print("   ⚠ Multiple topics dominant → Co-pathology detected")
            print("   This subject shows mixed pathology patterns")
        else:
            dominant_topic = results['top_topics'][0][0]
            print(f"   ✓ Topic {dominant_topic} dominant → Single pathology pattern")

    print(f"\n{'='*70}\n")


def infer_batch_subjects(
    model,
    X_new: np.ndarray,
    feature_names: List[str],
    dx_labels: List[str],
    subject_ids: List[str] = None
) -> pd.DataFrame:
    """
    Infer diagnosis and topics for multiple new subjects.

    Parameters
    ----------
    model : CoPathologySLDA
        Trained sLDA model
    X_new : np.ndarray, shape (n_subjects, n_features)
        Regional atrophy values
    feature_names : List[str]
        Feature names
    dx_labels : List[str]
        Diagnosis labels
    subject_ids : List[str], optional
        Subject identifiers

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns:
        - Subject_ID
        - Predicted_DX
        - DX_Confidence
        - Topic_0, Topic_1, ..., Topic_K (proportions)
        - DX_Prob_AD, DX_Prob_PD, ... (probabilities for each diagnosis)
    """
    n_subjects = X_new.shape[0]

    if subject_ids is None:
        subject_ids = [f"Subject_{i}" for i in range(n_subjects)]

    # Infer for all subjects
    all_results = []
    for i in range(n_subjects):
        result = infer_new_subject(
            model, X_new[i], feature_names, dx_labels, subject_ids[i]
        )
        all_results.append(result)

    # Create DataFrame
    data = {
        'Subject_ID': [r['subject_id'] for r in all_results],
        'Predicted_DX': [r['predicted_diagnosis'] for r in all_results],
        'DX_Confidence': [max(r['diagnosis_probs']) for r in all_results]
    }

    # Add topic proportions
    n_topics = len(all_results[0]['topic_mixture'])
    for k in range(n_topics):
        data[f'Topic_{k}'] = [r['topic_mixture'][k] for r in all_results]

    # Add diagnosis probabilities
    for dx in dx_labels:
        data[f'DX_Prob_{dx}'] = [r['diagnosis_breakdown'][dx] for r in all_results]

    results_df = pd.DataFrame(data)

    return results_df


def compare_subject_to_training(
    model,
    X_new: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dx_labels: List[str],
    subject_id: str = "New Subject"
) -> Dict:
    """
    Compare a new subject's topic mixture to training subjects.

    Parameters
    ----------
    model : CoPathologySLDA
        Trained sLDA model
    X_new : np.ndarray, shape (n_features,)
        New subject's atrophy values
    X_train : np.ndarray
        Training data features
    y_train : np.ndarray
        Training data diagnoses
    dx_labels : List[str]
        Diagnosis labels
    subject_id : str
        Subject identifier

    Returns
    -------
    comparison : dict
        Contains similarity scores to each diagnosis group
    """
    # Infer new subject's topic mixture
    theta_new = infer_topic_mixture(model, X_new.reshape(1, -1))[0]

    # Get training topic mixtures
    theta_train = model.get_patient_mixtures()

    # Compare to each diagnosis group
    similarities = {}
    for dx_id, dx_label in enumerate(dx_labels):
        # Get training subjects with this diagnosis
        dx_mask = (y_train == dx_id)
        if not dx_mask.any():
            continue

        # Compute cosine similarity to mean topic mixture of this group
        mean_theta_dx = theta_train[dx_mask].mean(axis=0)

        # Cosine similarity
        similarity = np.dot(theta_new, mean_theta_dx) / (
            np.linalg.norm(theta_new) * np.linalg.norm(mean_theta_dx)
        )

        similarities[dx_label] = similarity

    comparison = {
        'subject_id': subject_id,
        'topic_mixture': theta_new,
        'similarities': similarities,
        'most_similar_dx': max(similarities.items(), key=lambda x: x[1])[0]
    }

    return comparison


if __name__ == "__main__":
    print("Inference utilities for new subjects")
    print("=" * 70)
    print("\nExample usage:")
    print("""
    # Load trained model
    from slda_model import CoPathologySLDA
    import pickle

    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # New subject data (62 cortical features)
    X_new_subject = np.array([...])  # shape: (62,)

    # Infer diagnosis and topic mixture
    results = infer_new_subject(
        model,
        X_new_subject,
        feature_names,
        dx_labels,
        subject_id="SUBJ_NEW_001"
    )

    # Print results
    print_inference_results(results)

    # Access specific values
    print(f"Predicted diagnosis: {results['predicted_diagnosis']}")
    print(f"Topic mixture: {results['topic_mixture']}")
    print(f"Diagnosis probabilities: {results['diagnosis_breakdown']}")
    """)
