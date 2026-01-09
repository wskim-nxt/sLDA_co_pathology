# Supervised LDA for Co-Pathology: Mathematical Foundations

## Core Concept

**Goal:** Discover latent pathology patterns (topics) from brain atrophy data while simultaneously predicting diagnosis.

**Key Idea:** Each patient has a **mixture of pathology patterns** (not just one pure disease). sLDA discovers these patterns AND links them to clinical diagnoses.

---

## Why sLDA for Co-Pathology?

### Traditional Approach Problems
- **Single diagnosis assumption:** Each patient labeled as AD, PD, or DLB
- **Reality:** Many patients have **overlapping pathologies** (e.g., AD + vascular, DLB = AD + Lewy bodies)
- **Lost information:** Can't capture the continuum of mixed presentations

### sLDA Solution
- **Topic mixture:** Each patient = combination of multiple pathology patterns
- **Supervised learning:** Topics must be useful for predicting diagnosis
- **Interpretability:** Topics = regional atrophy profiles we can understand

**Example Patient:**
```
Patient_123:
  - 60% Topic 0 (limbic atrophy → AD-like)
  - 30% Topic 1 (cortical atrophy → DLB-like)
  - 10% Topic 2 (minimal atrophy → HC-like)

→ Mixed AD-DLB pathology captured!
```

---

## Mathematical Framework

### The Generative Story

Imagine how a patient's brain atrophy data is "generated":

1. **Nature picks pathology patterns** (topics exist in the world)
2. **Each patient gets a mixture** of these patterns (θ)
3. **Atrophy values emerge** from this mixture (X)
4. **Diagnosis follows** from the mixture (y)

---

## Core Variables

| Symbol | Name | Dimensions | Meaning |
|--------|------|------------|---------|
| **D** | Patients | 209 | Number of subjects |
| **V** | Features | 62 | Number of brain regions (cortical) |
| **K** | Topics | 4 | Number of pathology patterns |
| **C** | Classes | 5 | Number of diagnoses (AD, PD, DLB, SVAD, HC) |
| **β** | Topic patterns | K × V | What regions are affected in each pattern |
| **θ** | Patient mixtures | D × K | How much of each pattern each patient has |
| **η** | Diagnosis weights | K × C | How topics predict diagnoses |
| **X** | Atrophy data | D × V | Observed regional volumes |
| **y** | Diagnoses | D | Observed diagnosis labels |

---

## The sLDA Model (Full Math)

### 1. Topic Patterns (β)

**What:** Each topic defines a regional atrophy signature.

**Math:**
```
For each topic k = 1, ..., K:
  For each region v = 1, ..., V:
    β_kv ~ Normal(0, σ_β²)
```

**Code:** [slda_model.py:125-131](slda_model.py:125-131)
```python
# ---- Topic-specific atrophy patterns ----
# Each topic k has a mean atrophy value for each of V regions
# Shape: (K topics, V features)
beta = pm.Normal(
    "beta",
    mu=0.0,
    sigma=self.feature_prior_std,
    shape=(self.n_topics, self.n_features_)
)
```

**Interpretation:**
- `β[0, :]` = Topic 0's atrophy profile across 62 regions
- `β[0, 5] = +0.8` means region 5 has high atrophy in Topic 0
- `β[1, 5] = -0.3` means region 5 is relatively preserved in Topic 1

---

### 2. Patient Topic Mixtures (θ)

**What:** Each patient has a probability distribution over topics.

**Math:**
```
For each patient d = 1, ..., D:
  θ_d ~ Dirichlet(α)

where:
  θ_d = [θ_d1, θ_d2, ..., θ_dK]
  Σ_k θ_dk = 1  (proportions sum to 1)
  θ_dk ≥ 0      (non-negative)
```

**Code:** [slda_model.py:133-139](slda_model.py:133-139)
```python
# ---- Patient-topic mixtures ----
# Each patient d has a mixture over K topics
# Shape: (D patients, K topics)
# Constraint: each row sums to 1
theta = pm.Dirichlet(
    "theta",
    a=alpha,
    shape=(self.n_patients_, self.n_topics)
)
```

**The Dirichlet Distribution:**
- **Prior:** `α = [1, 1, 1, 1]` (uniform prior, no preference)
- **Effect:** Patient can have ANY mixture of topics
- **Example outputs:**
  - Pure: `[0.95, 0.02, 0.02, 0.01]` → mostly Topic 0
  - Mixed: `[0.40, 0.35, 0.20, 0.05]` → co-pathology!

---

### 3. Atrophy Likelihood (Continuous Features)

**What:** Patient's observed atrophy is a weighted combination of topic patterns.

**Math:**
```
For each patient d and region v:
  x_dv ~ Normal(μ_dv, σ_x²)

where:
  μ_dv = Σ_k θ_dk × β_kv = (θ_d ⊗ β)_v
```

**Code:** [slda_model.py:141-158](slda_model.py:141-158)
```python
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
```

**Key Difference from Text LDA:**
- **Text LDA:** Words are discrete counts → Multinomial likelihood
- **Our sLDA:** Atrophy is continuous → **Normal likelihood**

**Intuition:**
```
Patient with θ = [0.6, 0.4, 0.0, 0.0]:

  Expected atrophy in region v:
  μ_v = 0.6 × β[0,v] + 0.4 × β[1,v]
      = 0.6 × (limbic pattern) + 0.4 × (cortical pattern)

  Observed:
  x_v ~ Normal(μ_v, σ_x)
```

---

### 4. Supervised Component (Diagnosis Prediction)

**What:** Topics predict diagnosis via softmax regression.

**Math:**
```
For each patient d:
  logit_dc = Σ_k θ_dk × η_kc

  P(y_d = c | θ_d) = exp(logit_dc) / Σ_c' exp(logit_dc')  [Softmax]

  y_d ~ Categorical(P(y_d | θ_d))
```

**Code:** [slda_model.py:160-175](slda_model.py:160-175)
```python
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
```

**Interpretation of η:**
```
η = [
  # AD   PD   DLB  SVAD  HC
  [+2.3, -0.5, +0.8, +1.2, -3.1],  # Topic 0 (limbic)
  [+0.9, -0.3, +2.1, +0.4, -2.5],  # Topic 1 (cortical)
  [-0.5, +1.8, +0.6, -0.2, -1.2],  # Topic 2 (frontal)
  [-2.8, -1.5, -2.3, -2.1, +4.5],  # Topic 3 (preservation)
]

Topic 0 → strongly predicts AD (+2.3)
Topic 3 → strongly predicts HC (+4.5)
```

**Why This Works:**
- **Force topics to be useful:** They must predict diagnosis
- **But allow flexibility:** Topics can predict multiple diagnoses
- **Captures overlap:** DLB-predicting topic might also help predict AD

---

## Complete Model Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATIVE PROCESS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Nature creates K pathology patterns:                    │
│     β ~ Normal(0, σ²)                    [K × V matrix]     │
│                                                              │
│  2. Each patient gets a topic mixture:                      │
│     θ_d ~ Dirichlet(α)                   [K-dim vector]     │
│                                                              │
│  3. Atrophy emerges from mixture:                           │
│     x_dv ~ Normal(θ_d @ β_v, σ_x²)       [D × V matrix]     │
│                                                              │
│  4. Diagnosis follows from mixture:                         │
│     y_d ~ Categorical(Softmax(θ_d @ η))  [D-dim vector]     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Inference: Learning from Data

**Given:** Observed data (X, y)
**Learn:** Hidden variables (β, θ, η)

**Method:** Bayesian inference via MCMC (NUTS sampler)

**Code:** [slda_model.py:177-185](slda_model.py:177-185)
```python
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
```

**What MCMC Does:**
1. Starts with random β, θ, η
2. Proposes changes that increase P(β, θ, η | X, y)
3. Samples from posterior distribution
4. Returns uncertainty: not just point estimates!

**Output:**
- `posterior["beta"]`: 2000 samples × 4 chains × (K × V) matrix
- We use the mean: `β̂ = mean(posterior["beta"])`

---

## Key Mathematical Properties

### 1. Identifiability
**Problem:** β and θ are not uniquely identified (label switching).
**Solution:** Supervised component (η) breaks symmetry by linking topics to diagnoses.

### 2. Topic Mixtures Sum to 1
```python
theta = pm.Dirichlet(...)  # Ensures Σ_k θ_dk = 1
```
This makes θ interpretable as proportions.

### 3. Normal Likelihood Choice
**Why Normal?**
- Atrophy values are continuous (z-scored volumes)
- Already normalized/standardized
- Symmetric around zero

**Alternative:** Could use Lognormal if values are strictly positive and right-skewed.

---

## Example Walkthrough

Let's trace through one patient:

**Patient 42:**
- Observed: `X[42, :] = [0.5, 0.8, -0.2, ..., 0.3]` (62 values)
- Diagnosis: `y[42] = 0` (AD)

**Step 1: Model learns topics**
```python
β[0, :] = [0.9, 0.7, 0.1, ..., 0.4]  # Limbic pattern
β[1, :] = [0.2, 0.3, 0.8, ..., 0.6]  # Cortical pattern
β[2, :] = [-0.1, 0.1, 0.4, ..., 0.2] # Frontal pattern
β[3, :] = [-0.5, -0.4, -0.2, ..., -0.3] # Preservation
```

**Step 2: Model infers mixture**
```python
θ[42, :] = [0.65, 0.25, 0.08, 0.02]
# 65% limbic, 25% cortical, 8% frontal, 2% preservation
```

**Step 3: Predicted atrophy**
```python
μ[42, :] = θ[42, :] @ β
         = 0.65 × β[0,:] + 0.25 × β[1,:] + 0.08 × β[2,:] + 0.02 × β[3,:]

For region 0:
μ[42, 0] = 0.65 × 0.9 + 0.25 × 0.2 + 0.08 × (-0.1) + 0.02 × (-0.5)
         = 0.585 + 0.05 - 0.008 - 0.01
         = 0.617

Observed: X[42, 0] = 0.5
Likelihood: P(X[42,0] | ...) = Normal(0.5 | μ=0.617, σ=0.1)
```

**Step 4: Predicted diagnosis**
```python
logits[42, :] = θ[42, :] @ η
              = 0.65 × η[0,:] + 0.25 × η[1,:] + ...

For AD (class 0):
logits[42, 0] = 0.65 × 2.3 + 0.25 × 0.9 + 0.08 × (-0.5) + 0.02 × (-2.8)
              = 1.495 + 0.225 - 0.04 - 0.056
              = 1.624

P(AD | θ[42]) = exp(1.624) / (exp(1.624) + exp(-0.3) + ... + exp(-2.1))
              ≈ 0.72  (72% probability)

True diagnosis: AD ✓
```

---

## Why This Model Captures Co-Pathology

### Traditional Classification
```
Input: X → Model → Output: "AD" or "PD" or "DLB"
```
**Problem:** Binary decision, no mixed pathology information.

### Our sLDA Model
```
Input: X → Model → Output:
  - θ = [0.65, 0.25, 0.08, 0.02]  ← Co-pathology mixture!
  - P(AD) = 0.72, P(DLB) = 0.18, ...  ← Uncertainty
  - Predicted: AD
```

**Benefits:**
1. **Mixture visible:** We see 65% AD-like + 25% DLB-like
2. **Interpretable:** Topics are regional atrophy patterns
3. **Uncertainty:** Probability distribution over diagnoses
4. **Clinical utility:** "This patient has mixed AD-DLB pathology"

---

## Comparison: Text sLDA vs. Our sLDA

| Aspect | Text sLDA | Our Co-Pathology sLDA |
|--------|-----------|------------------------|
| **Documents** | Movie reviews | Patients |
| **Words** | Vocabulary words | Brain regions |
| **Word values** | Counts (discrete) | Atrophy (continuous) |
| **Topics** | Themes (e.g., "action", "romance") | Pathology patterns (e.g., "limbic", "cortical") |
| **Response** | Sentiment (1-5 stars) | Diagnosis (AD, PD, etc.) |
| **Likelihood** | Multinomial | **Normal** |
| **Supervised** | Linear regression | **Softmax classification** |
| **Topic mixture** | Document about multiple themes | **Patient with multiple pathologies** |

**Key Innovation:** Adapted LDA for continuous neuroimaging + categorical diagnosis.

---

## References

### Original sLDA Paper
Blei, D. M., & McAuliffe, J. D. (2007). **Supervised topic models**. *Advances in Neural Information Processing Systems*, 20.

**Our adaptation:**
- Continuous likelihood for brain atrophy (Normal vs Multinomial)
- Categorical outcome for diagnosis (Softmax vs Linear)
- Applied to co-pathology discovery in neurodegenerative diseases

---

## Summary

**sLDA discovers latent pathology patterns by:**
1. **Decomposing** patient atrophy into topic mixtures (θ)
2. **Learning** what regions define each pattern (β)
3. **Linking** patterns to diagnoses (η)
4. **Capturing** co-pathology through mixed topic membership

**Math in one line:**
```
P(X, y | β, θ, η) = ∏_d [ ∏_v Normal(x_dv | θ_d @ β_v, σ²) × Categorical(y_d | Softmax(θ_d @ η)) ]
```

**Clinical insight:**
> "This patient has 60% AD-like limbic atrophy and 30% DLB-like cortical atrophy, explaining their mixed clinical presentation."

That's the power of sLDA for co-pathology analysis! 🧠
