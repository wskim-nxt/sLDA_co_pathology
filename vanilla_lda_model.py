"""
Vanilla (Unsupervised) LDA model for discovering atrophy patterns.

This module implements a standard LDA model adapted for continuous neuroimaging
features WITHOUT supervision from diagnosis labels. The topics discovered are
purely data-driven atrophy patterns.
"""

import numpy as np
import pymc as pm
from typing import Optional, List, Tuple


class VanillaLDA:
    """
    Unsupervised LDA for atrophy pattern discovery.

    This model discovers latent atrophy patterns (topics) from regional
    gray matter values WITHOUT using diagnosis labels during training.

    Parameters
    ----------
    n_topics : int, default=4
        Number of latent atrophy patterns to discover
    alpha_prior : float, default=1.0
        Concentration parameter for Dirichlet prior on patient topic mixtures
        - alpha > 1: patients have more uniform topic mixtures
        - alpha < 1: patients are sparse (dominated by few topics)
        - alpha = 1: uniform prior
    feature_prior_std : float, default=1.0
        Standard deviation for prior on topic-feature associations
    random_state : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    trace_ : arviz.InferenceData
        Posterior samples from inference
    n_patients_ : int
        Number of patients in training data
    n_features_ : int
        Number of features
    """

    def __init__(
        self,
        n_topics: int = 4,
        alpha_prior: float = 1.0,
        feature_prior_std: float = 1.0,
        random_state: int = 42
    ):
        self.n_topics = n_topics
        self.alpha_prior = alpha_prior
        self.feature_prior_std = feature_prior_std
        self.random_state = random_state

        # Will be set during fit
        self.trace_ = None
        self.model_ = None
        self.approx_ = None
        self.inference_method_ = None
        self.n_patients_ = None
        self.n_features_ = None

    def fit(
        self,
        X: np.ndarray,
        inference: str = "advi",
        n_samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        n_advi_iterations: int = 30000,
        **kwargs
    ):
        """
        Fit the unsupervised LDA model to patient data.

        Parameters
        ----------
        X : np.ndarray, shape (n_patients, n_features)
            Regional atrophy values for each patient
        inference : str, default="advi"
            Inference method: "mcmc" for NUTS sampling, "advi" for variational inference
        n_samples : int, default=2000
            Number of samples to draw
        tune : int, default=1000
            Number of tuning samples (only for mcmc)
        chains : int, default=4
            Number of parallel chains (only for mcmc)
        target_accept : float, default=0.9
            Target acceptance rate (only for mcmc)
        n_advi_iterations : int, default=30000
            Number of ADVI iterations (only for advi)

        Returns
        -------
        self : VanillaLDA
            Fitted model
        """
        self.n_patients_, self.n_features_ = X.shape
        self.inference_method_ = inference.lower()

        if self.inference_method_ not in ["mcmc", "advi"]:
            raise ValueError(f"inference must be 'mcmc' or 'advi', got '{inference}'")

        print(f"Fitting Vanilla LDA model (unsupervised):")
        print(f"  Patients: {self.n_patients_}")
        print(f"  Features: {self.n_features_}")
        print(f"  Topics: {self.n_topics}")
        if self.inference_method_ == "mcmc":
            print(f"  Inference: MCMC (NUTS)")
            print(f"  Sampling: {n_samples} samples × {chains} chains")
        else:
            print(f"  Inference: ADVI (Variational)")
            print(f"  Iterations: {n_advi_iterations}, then {n_samples} samples")

        with pm.Model() as self.model_:
            # Dirichlet concentration for patient-topic mixtures
            alpha = self.alpha_prior * np.ones(self.n_topics)

            # ---- Topic-specific atrophy patterns ----
            # Each topic k has a mean atrophy value for each of V regions
            # Shape: (K topics, V features)
            beta = pm.Normal(
                "beta",
                mu=0.0,
                sigma=self.feature_prior_std,
                shape=(self.n_topics, self.n_features_)
            )

            # ---- Patient-topic mixtures ----
            # Each patient d has a mixture over K topics
            # Shape: (D patients, K topics)
            theta = pm.Dirichlet(
                "theta",
                a=alpha,
                shape=(self.n_patients_, self.n_topics)
            )

            # ---- Likelihood for regional atrophy values ----
            # For each patient, observed atrophy is a mixture of topic patterns
            # x_d ~ Normal(theta_d @ beta, sigma)

            # Compute expected atrophy for each patient-region pair
            mu_x = pm.math.dot(theta, beta)

            # Observation noise
            sigma_x = pm.HalfNormal("sigma_x", 1.0)

            # Observed atrophy values
            x_obs = pm.Normal(
                "x_obs",
                mu=mu_x,
                sigma=sigma_x,
                observed=X
            )

            # ---- Inference ----
            if self.inference_method_ == "mcmc":
                print("\nStarting MCMC sampling...")
                self.trace_ = pm.sample(
                    draws=n_samples,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=self.random_state,
                    return_inferencedata=True,
                    **kwargs
                )
                print("\nSampling complete!")
                self._print_convergence_diagnostics()
            else:
                print("\nStarting ADVI optimization...")
                self.approx_ = pm.fit(
                    n=n_advi_iterations,
                    method="advi",
                    random_seed=self.random_state,
                    **kwargs
                )
                print(f"\nADVI optimization complete! Final ELBO: {self.approx_.hist[-1]:.2f}")

                print(f"Drawing {n_samples} samples from approximate posterior...")
                self.trace_ = self.approx_.sample(n_samples)

                self._print_advi_diagnostics()

        return self

    def _print_convergence_diagnostics(self):
        """Print convergence diagnostics for MCMC."""
        import arviz as az

        rhat = az.rhat(self.trace_)

        print("\nConvergence diagnostics (R-hat):")
        for var in ["beta", "theta", "sigma_x"]:
            if var in rhat:
                rhat_vals = rhat[var].values.flatten()
                max_rhat = np.max(rhat_vals)
                mean_rhat = np.mean(rhat_vals)
                print(f"  {var}: mean={mean_rhat:.4f}, max={max_rhat:.4f}", end="")
                if max_rhat > 1.01:
                    print(" (Warning: Poor convergence)")
                else:
                    print(" (OK)")

    def _print_advi_diagnostics(self):
        """Print diagnostics for ADVI inference."""
        print("\nADVI Diagnostics:")

        elbo_history = self.approx_.hist
        final_elbo = elbo_history[-1]
        elbo_change = elbo_history[-1] - elbo_history[-100] if len(elbo_history) >= 100 else 0

        print(f"  Final ELBO: {final_elbo:.2f}")
        print(f"  ELBO change (last 100 iters): {elbo_change:.2f}")

        if abs(elbo_change) > 10:
            print("  (Warning: ELBO may not have converged - consider more iterations)")
        else:
            print("  (OK - ELBO appears stable)")

    def plot_elbo(self, figsize=(10, 4), save_path=None):
        """Plot ELBO convergence for ADVI."""
        if self.approx_ is None:
            raise ValueError("ELBO plot only available for ADVI inference.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.approx_.hist, alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ELBO")
        ax.set_title("ADVI Convergence (Evidence Lower Bound)")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_topic_patterns(self):
        """
        Get the mean topic-region association matrix.

        Returns
        -------
        beta_mean : np.ndarray, shape (n_topics, n_features)
            Mean posterior atrophy pattern for each topic
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        beta_mean = self.trace_.posterior["beta"].mean(dim=["chain", "draw"]).values
        return beta_mean

    def get_patient_mixtures(self):
        """
        Get the mean patient-topic mixture matrix.

        Returns
        -------
        theta_mean : np.ndarray, shape (n_patients, n_topics)
            Mean posterior topic proportions for each patient
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        theta_mean = self.trace_.posterior["theta"].mean(dim=["chain", "draw"]).values
        return theta_mean

    def get_topic_top_regions(
        self,
        topic_id: int,
        feature_names: list,
        n_regions: int = 10,
        absolute: bool = True
    ):
        """
        Get the top regions associated with a specific topic.

        Parameters
        ----------
        topic_id : int
            Topic index (0 to n_topics-1)
        feature_names : list
            Names of features
        n_regions : int, default=10
            Number of top regions to return
        absolute : bool, default=True
            If True, rank by absolute value

        Returns
        -------
        top_regions : list of tuples
            List of (region_name, weight) tuples
        """
        beta = self.get_topic_patterns()

        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"topic_id must be between 0 and {self.n_topics-1}")

        topic_weights = beta[topic_id, :]

        if absolute:
            sorted_idx = np.argsort(np.abs(topic_weights))[::-1]
        else:
            sorted_idx = np.argsort(topic_weights)[::-1]

        top_regions = [
            (feature_names[idx], topic_weights[idx])
            for idx in sorted_idx[:n_regions]
        ]

        return top_regions

    def infer_new_patient_mixtures(self, X_new: np.ndarray):
        """
        Infer topic mixtures for new patients using least squares.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new_patients, n_features)
            Regional atrophy values for new patients

        Returns
        -------
        theta_new : np.ndarray, shape (n_new_patients, n_topics)
            Inferred topic proportions
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        from scipy.linalg import lstsq

        beta = self.get_topic_patterns()

        # Solve: X_new ≈ theta_new @ beta
        theta_new, _, _, _ = lstsq(beta.T, X_new.T)
        theta_new = theta_new.T

        # Ensure non-negative and normalize
        theta_new = np.maximum(theta_new, 0)
        theta_new = theta_new / theta_new.sum(axis=1, keepdims=True)

        return theta_new


if __name__ == "__main__":
    print("Testing VanillaLDA with synthetic data...\n")

    np.random.seed(42)

    # Generate synthetic data
    n_patients = 50
    n_features = 20
    true_topics = 3

    # Create topic patterns
    true_beta = np.random.randn(true_topics, n_features)

    # Create patient topic mixtures
    from scipy.stats import dirichlet
    true_theta = dirichlet.rvs([1] * true_topics, size=n_patients)

    # Generate atrophy data
    X_synth = true_theta @ true_beta + 0.3 * np.random.randn(n_patients, n_features)

    print(f"Synthetic data: {n_patients} patients, {n_features} features")

    # Fit model with ADVI (faster for testing)
    model = VanillaLDA(n_topics=3, alpha_prior=1.0)
    model.fit(X_synth, inference='advi', n_advi_iterations=10000, n_samples=500)

    # Check outputs
    print("\nModel outputs:")
    print(f"Topic patterns shape: {model.get_topic_patterns().shape}")
    print(f"Patient mixtures shape: {model.get_patient_mixtures().shape}")

    print("\n✓ VanillaLDA test complete!")
