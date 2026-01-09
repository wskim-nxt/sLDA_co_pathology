"""
Supervised Latent Dirichlet Allocation model for co-pathology analysis.

This module implements an sLDA model adapted for continuous neuroimaging features
with categorical diagnosis outcomes.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Optional, Tuple
import warnings


class CoPathologySLDA:
    """
    Supervised LDA for co-pathology pattern discovery.

    This model discovers latent pathology patterns (topics) from regional
    gray matter atrophy values and links them to diagnosis outcomes.

    Parameters
    ----------
    n_topics : int, default=4
        Number of latent pathology patterns to discover
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
        Posterior samples from MCMC inference
    n_patients_ : int
        Number of patients in training data
    n_features_ : int
        Number of cortical features
    n_classes_ : int
        Number of diagnosis categories
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
        self.n_patients_ = None
        self.n_features_ = None
        self.n_classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        **kwargs
    ):
        """
        Fit the supervised LDA model to patient data.

        Parameters
        ----------
        X : np.ndarray, shape (n_patients, n_features)
            Regional atrophy values for each patient
        y : np.ndarray, shape (n_patients,)
            Integer-encoded diagnosis labels (0 to n_classes-1)
        n_samples : int, default=2000
            Number of MCMC samples to draw per chain
        tune : int, default=1000
            Number of tuning (burn-in) samples
        chains : int, default=4
            Number of parallel MCMC chains
        target_accept : float, default=0.9
            Target acceptance rate for NUTS sampler
        **kwargs
            Additional arguments passed to pm.sample()

        Returns
        -------
        self : CoPathologySLDA
            Fitted model
        """
        self.n_patients_, self.n_features_ = X.shape
        self.n_classes_ = len(np.unique(y))

        print(f"Fitting sLDA model:")
        print(f"  Patients: {self.n_patients_}")
        print(f"  Features: {self.n_features_}")
        print(f"  Topics: {self.n_topics}")
        print(f"  Diagnoses: {self.n_classes_}")
        print(f"  Sampling: {n_samples} samples × {chains} chains")

        with pm.Model() as self.model_:
            # ---- Hyperpriors ----
            # Dirichlet concentration for patient-topic mixtures
            # alpha = pm.Constant("alpha", self.alpha_prior * np.ones(self.n_topics))
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
            # Constraint: each row sums to 1
            theta = pm.Dirichlet(
                "theta",
                a=alpha,
                shape=(self.n_patients_, self.n_topics)
            )

            # ---- Likelihood for regional atrophy values ----
            # For each patient, observed atrophy is a mixture of topic patterns
            # x_d ~ Normal(theta_d @ beta, sigma)
            # This creates a weighted combination of topic patterns

            # Compute expected atrophy for each patient-region pair
            # Shape: (D patients, V features)
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

            # ---- Supervised component: Topic → Diagnosis ----
            # Linear combination of topics predicts diagnosis
            # Shape: (K topics, C classes)
            eta = pm.Normal(
                "eta",
                mu=0.0,
                sigma=2.0,
                shape=(self.n_topics, self.n_classes_)
            )

            # For each patient, compute class logits from topic mixture
            # Shape: (D patients, C classes)
            logits = pm.math.dot(theta, eta)

            # Categorical likelihood for diagnosis
            y_obs = pm.Categorical(
                "y_obs",
                logit_p=logits,
                observed=y
            )

            # ---- MCMC Sampling ----
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

        return self

    def _print_convergence_diagnostics(self):
        """Print convergence diagnostics for key parameters."""
        import arviz as az

        # Check R-hat for convergence
        rhat = az.rhat(self.trace_)

        print("\nConvergence diagnostics (R-hat):")
        for var in ["beta", "theta", "eta", "sigma_x"]:
            if var in rhat:
                rhat_vals = rhat[var].values.flatten()
                max_rhat = np.max(rhat_vals)
                mean_rhat = np.mean(rhat_vals)
                print(f"  {var}: mean={mean_rhat:.4f}, max={max_rhat:.4f}", end="")
                if max_rhat > 1.01:
                    print(" ⚠ Warning: Poor convergence")
                else:
                    print(" ✓")

    def get_topic_patterns(self):
        """
        Get the mean topic-region association matrix.

        Returns
        -------
        beta_mean : np.ndarray, shape (n_topics, n_features)
            Mean posterior atrophy pattern for each topic
            Higher values indicate stronger association
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
            Each row sums to 1
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        theta_mean = self.trace_.posterior["theta"].mean(dim=["chain", "draw"]).values
        return theta_mean

    def get_diagnosis_weights(self):
        """
        Get the mean topic-diagnosis association matrix.

        Returns
        -------
        eta_mean : np.ndarray, shape (n_topics, n_classes)
            Mean posterior weights linking topics to diagnoses
            Higher values indicate stronger association
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        eta_mean = self.trace_.posterior["eta"].mean(dim=["chain", "draw"]).values
        return eta_mean

    def get_topic_top_regions(
        self,
        topic_id: int,
        feature_names: list,
        n_regions: int = 10,
        absolute: bool = True
    ) -> list:
        """
        Get the top regions associated with a specific topic.

        Parameters
        ----------
        topic_id : int
            Topic index (0 to n_topics-1)
        feature_names : list
            Names of cortical features
        n_regions : int, default=10
            Number of top regions to return
        absolute : bool, default=True
            If True, rank by absolute value (both positive and negative)
            If False, rank by actual value (highest values)

        Returns
        -------
        top_regions : list of tuples
            List of (region_name, weight) tuples, sorted by importance
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

    def predict_diagnosis_proba(self, X_new: np.ndarray):
        """
        Predict diagnosis probabilities for new patients.

        Note: This uses the posterior mean parameters, not full uncertainty.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new_patients, n_features)
            Regional atrophy values for new patients

        Returns
        -------
        probs : np.ndarray, shape (n_new_patients, n_classes)
            Predicted diagnosis probabilities
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        warnings.warn(
            "Prediction uses point estimates (posterior means) and does not "
            "account for full posterior uncertainty. For proper Bayesian "
            "prediction, use posterior predictive sampling.",
            UserWarning
        )

        # Get posterior means
        beta = self.get_topic_patterns()  # (K, V)
        eta = self.get_diagnosis_weights()  # (K, C)

        # Infer topic mixtures for new patients
        # This is a simplified approach - ideally would use variational inference
        # For now, we solve: X_new ≈ theta_new @ beta
        # Using least squares: theta_new = X_new @ beta.T @ inv(beta @ beta.T)

        from scipy.linalg import lstsq

        # Solve for topic mixtures (constrained to sum to 1 is approximated)
        theta_new, _, _, _ = lstsq(beta.T, X_new.T)
        theta_new = theta_new.T  # (N_new, K)

        # Ensure non-negative and normalize to sum to 1
        theta_new = np.maximum(theta_new, 0)
        theta_new = theta_new / theta_new.sum(axis=1, keepdims=True)

        # Compute logits and convert to probabilities
        logits = theta_new @ eta  # (N_new, C)
        probs = self._softmax(logits)

        return probs

    def predict_diagnosis(self, X_new: np.ndarray):
        """
        Predict diagnosis class for new patients.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new_patients, n_features)
            Regional atrophy values for new patients

        Returns
        -------
        y_pred : np.ndarray, shape (n_new_patients,)
            Predicted diagnosis class (0 to n_classes-1)
        """
        probs = self.predict_diagnosis_proba(X_new)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _softmax(x: np.ndarray):
        """Compute softmax in a numerically stable way."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    # Simple test with synthetic data
    print("Testing CoPathologySLDA with synthetic data...\n")

    np.random.seed(42)

    # Generate synthetic data
    n_patients = 50
    n_features = 20
    n_classes = 3
    true_topics = 3

    # Create topic patterns
    true_beta = np.random.randn(true_topics, n_features)

    # Create patient topic mixtures
    from scipy.stats import dirichlet
    true_theta = dirichlet.rvs([1] * true_topics, size=n_patients)

    # Generate atrophy data
    X_synth = true_theta @ true_beta + 0.3 * np.random.randn(n_patients, n_features)

    # Generate diagnoses (correlated with topics)
    topic_dx_weights = np.random.randn(true_topics, n_classes)
    logits = true_theta @ topic_dx_weights
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_synth = np.array([np.random.choice(n_classes, p=p) for p in probs])

    print(f"Synthetic data: {n_patients} patients, {n_features} features, {n_classes} classes")
    print(f"Diagnosis distribution: {np.bincount(y_synth)}\n")

    # Fit model
    model = CoPathologySLDA(n_topics=3, alpha_prior=1.0)
    model.fit(X_synth, y_synth, n_samples=500, tune=500, chains=2)

    # Check outputs
    print("\nModel outputs:")
    print(f"Topic patterns shape: {model.get_topic_patterns().shape}")
    print(f"Patient mixtures shape: {model.get_patient_mixtures().shape}")
    print(f"Diagnosis weights shape: {model.get_diagnosis_weights().shape}")

    print("\n✓ Model test complete!")
