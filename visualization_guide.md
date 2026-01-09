# Topic Composition Visualization Guide

This guide shows the **4 new visualization functions** added to help you understand what each topic represents.

## New Visualization Functions

### 1. `plot_topic_wordcloud_style()` ⭐ **Best for Quick Overview**

**Shows:** Top regions for each topic with horizontal bars, colored by sign (positive/negative).

```python
from visualization import plot_topic_wordcloud_style

plot_topic_wordcloud_style(
    topic_patterns,
    feature_names,
    n_top_regions=20,
    save_path='topic_composition.png'
)
```

**Output:**
```
Topic 0: Top 20 Regions
  ████████████ L_entorhinal         +0.452
  ██████████   R_parahippocampal    +0.398
  ████████     L_fusiform           +0.312
  ███          L_precuneus          +0.124
  ██           R_temporal           +0.089
  █            L_parietal           -0.056
  ████         R_frontal            -0.183
```

**Use when:** You want a clear, detailed list of which regions compose each topic.

---

### 2. `plot_topic_summary()` ⭐ **Best for Single Topic Deep Dive**

**Shows:** Everything about ONE topic:
- Top regions overall
- Left hemisphere breakdown
- Right hemisphere breakdown
- Which diagnoses it predicts
- Distribution across patients

```python
from visualization import plot_topic_summary

plot_topic_summary(
    topic_patterns,
    feature_names,
    diagnosis_weights,
    dx_labels,
    patient_mixtures=patient_mixtures,  # Optional
    diagnoses=y,                         # Optional
    topic_id=0,                          # Which topic to analyze
    save_path='topic_0_summary.png'
)
```

**Output:** A comprehensive 5-panel figure showing:
- Panel 1: Top 15 regions
- Panel 2: Top 10 left hemisphere regions
- Panel 3: Top 10 right hemisphere regions
- Panel 4: Association with each diagnosis
- Panel 5: How common this topic is in each diagnosis group

**Use when:** You want to deeply understand what a specific topic represents.

---

### 3. `plot_topic_comparison()` **Best for Comparing Topics**

**Shows:** Side-by-side comparison of multiple topics showing which regions differentiate them.

```python
from visualization import plot_topic_comparison

plot_topic_comparison(
    topic_patterns,
    feature_names,
    topic_ids=[0, 1, 2],  # Compare these topics
    n_top_regions=15,
    save_path='topic_comparison.png'
)
```

**Output:** Grouped bar chart showing:
```
               Topic 0   Topic 1   Topic 2
L_entorhinal   █████      █         ░
R_hippocampus  ████       ██        ░
L_precuneus    ░          █████     ███
L_frontal      ░          ██        █████
```

**Use when:** You want to see how topics differ from each other - which regions are unique to each pattern.

---

### 4. `plot_topic_composition_radar()` **Best for Visual Pattern Recognition**

**Shows:** Radar/spider charts showing topic "shape" across brain regions.

```python
from visualization import plot_topic_composition_radar

plot_topic_composition_radar(
    topic_patterns,
    feature_names,
    n_top_regions=15,
    save_path='topic_radar.png'
)
```

**Output:** Polar plots showing each topic as a distinctive "shape":
```
        L_temporal
            │
L_parietal ─┼─ R_temporal
            │
        L_frontal
```

**Use when:** You want an intuitive visual of each topic's regional profile at a glance.

---

## Complete Example Usage

```python
# After training your model
from preprocessing import prepare_slda_inputs
from slda_model import CoPathologySLDA
from visualization import (
    plot_topic_wordcloud_style,
    plot_topic_summary,
    plot_topic_comparison,
    plot_topic_composition_radar
)

# Get model results
topic_patterns = model.get_topic_patterns()
patient_mixtures = model.get_patient_mixtures()
diagnosis_weights = model.get_diagnosis_weights()

# 1. Quick overview of ALL topics
plot_topic_wordcloud_style(
    topic_patterns,
    feature_names,
    n_top_regions=20
)
plt.show()

# 2. Deep dive into Topic 0 (e.g., AD pattern)
plot_topic_summary(
    topic_patterns,
    feature_names,
    diagnosis_weights,
    dx_labels,
    patient_mixtures,
    y,
    topic_id=0
)
plt.show()

# 3. Compare Topic 0 vs Topic 1 (e.g., AD vs PD)
plot_topic_comparison(
    topic_patterns,
    feature_names,
    topic_ids=[0, 1]
)
plt.show()

# 4. See all topic "shapes" at once
plot_topic_composition_radar(
    topic_patterns,
    feature_names
)
plt.show()
```

---

## Interpreting the Visualizations

### Color Coding

- **Red bars** = Positive weights = Higher atrophy in this topic
- **Blue bars** = Negative weights = Lower atrophy (preservation) in this topic
- **Green (diagnosis)** = Positive association = Topic predicts this diagnosis
- **Red (diagnosis)** = Negative association = Topic counter-indicates this diagnosis

### What to Look For

**For a Topic to be "Limbic/AD-like":**
- High positive weights: entorhinal, parahippocampal, fusiform, temporal
- Positive association with AD diagnosis
- Common in AD patients, rare in HC patients

**For a Topic to be "Cortical/DLB-like":**
- High positive weights: precuneus, parietal, posterior cingulate
- Positive association with DLB (maybe also AD)
- Mixed distribution across patient groups

**For a Topic to be "PD-like":**
- Lower overall cortical atrophy
- Less limbic involvement
- Positive association with PD diagnosis

**For a Topic to be "Healthy/HC-like":**
- Very low or negative weights (preservation)
- Positive association with HC
- High proportion in HC patients, low in diseased groups

---

## Recommended Workflow

### Step 1: Start with Wordcloud Style
```python
plot_topic_wordcloud_style(topic_patterns, feature_names)
```
**Goal:** Get a quick sense of what regions define each topic.

### Step 2: Deep Dive Each Topic
```python
for topic_id in range(model.n_topics):
    plot_topic_summary(topic_patterns, feature_names,
                      diagnosis_weights, dx_labels,
                      patient_mixtures, y, topic_id=topic_id)
    plt.show()
```
**Goal:** Understand what each topic represents (limbic? cortical? subcortical?).

### Step 3: Compare Related Topics
```python
# Compare AD-like topics
plot_topic_comparison(topic_patterns, feature_names, topic_ids=[0, 2])
```
**Goal:** Understand how co-pathology patterns differ.

### Step 4: Print Top Regions
```python
for topic_id in range(model.n_topics):
    print(f"\nTopic {topic_id}:")
    top_regions = model.get_topic_top_regions(topic_id, feature_names, n_regions=10)
    for region, weight in top_regions:
        print(f"  {region:40s} {weight:+.3f}")
```
**Goal:** Get exact numerical values for publication/reporting.

---

## Example Output Interpretation

Let's say you have 4 topics and see:

**Topic 0:**
- Top regions: L_entorhinal, R_parahippocampal, L_fusiform, R_temporal
- Diagnosis: Strong positive for AD (+2.3), positive for SVAD (+1.1)
- **Interpretation:** **Limbic/medial temporal atrophy pattern** - classic AD pathology

**Topic 1:**
- Top regions: L_precuneus, R_parietal, L_posteriorcingulate
- Diagnosis: Positive for DLB (+1.8), moderate for AD (+0.7)
- **Interpretation:** **Posterior cortical atrophy** - common in DLB and AD

**Topic 2:**
- Top regions: L_frontal, R_frontal, L_precentral
- Diagnosis: Positive for PD (+1.5), small negative for AD (-0.3)
- **Interpretation:** **Frontal pattern** - PD-related or subcortical pattern

**Topic 3:**
- Top regions: All low/negative weights
- Diagnosis: Strong positive for HC (+3.1), negative for all diseases
- **Interpretation:** **Preservation pattern** - healthy brain signature

**Co-pathology Example:**
A patient with:
- 40% Topic 0 (limbic)
- 45% Topic 1 (cortical)
- 10% Topic 2 (frontal)
- 5% Topic 3 (healthy)

Shows **mixed AD-DLB pathology** with both limbic and cortical involvement!

---

## Tips

1. **Always visualize ALL topics** - even unexpected patterns can be informative
2. **Cross-reference with diagnosis weights** - see which topics predict which diseases
3. **Look at patient distributions** - does a "PD topic" actually appear in PD patients?
4. **Compare hemispheres** - is atrophy lateralized?
5. **Use multiple visualizations** - each shows different aspects of the same data

---

## Summary: Which Visualization When?

| Scenario | Use This |
|----------|----------|
| "What regions define each topic?" | `plot_topic_wordcloud_style()` |
| "Tell me everything about Topic 0" | `plot_topic_summary()` |
| "How do Topic 0 and 1 differ?" | `plot_topic_comparison()` |
| "Show me visual patterns" | `plot_topic_composition_radar()` |
| "I want all the old visualizations too" | Use the original functions from README |

Happy visualizing! 🎨🧠
