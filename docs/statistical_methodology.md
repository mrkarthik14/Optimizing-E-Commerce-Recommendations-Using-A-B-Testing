# Statistical Methodology
## A/B Testing for E-Commerce Recommendations

**Document Version:** 1.0  
**Date:** January 2024  
**Author:** Data Science Team

---

## Table of Contents

1. [Overview](#overview)
2. [Sample Size Determination](#sample-size-determination)
3. [Validity Checks](#validity-checks)
4. [Primary Analysis](#primary-analysis)
5. [Segmentation Analysis](#segmentation-analysis)
6. [Variance Reduction](#variance-reduction)
7. [Bayesian Analysis](#bayesian-analysis)
8. [Sensitivity Analysis](#sensitivity-analysis)
9. [Multiple Testing](#multiple-testing)
10. [Assumptions and Limitations](#assumptions-and-limitations)

---

## 1. Overview

This document details the statistical methodology used to analyze the e-commerce recommendation A/B test. We employ both frequentist and Bayesian approaches to ensure robust inference.

### 1.1 Statistical Framework

**Paradigms:**
- **Frequentist:** Classical hypothesis testing with p-values and confidence intervals
- **Bayesian:** Posterior probability that treatment is superior

**Why both?**
- Frequentist: Industry standard, easy to interpret, controls error rates
- Bayesian: Direct probability statements, incorporates prior knowledge, better for decision-making

### 1.2 Analysis Philosophy

**Principles:**
1. **Pre-specification:** All analyses planned before seeing data
2. **Transparency:** Document all decisions, no p-hacking
3. **Conservatism:** User-level analysis, two-tailed tests (where appropriate)
4. **Validity:** Extensive checks (SRM, balance, temporal stability)
5. **Practical significance:** Statistical significance necessary but not sufficient

---

## 2. Sample Size Determination

### 2.1 Power Analysis

We use the standard formula for two-sample proportion tests:

```
n = 2 * ((Z_α + Z_β)² * p_pooled * (1 - p_pooled)) / δ²
```

**Where:**
- `Z_α` = Z-score for significance level α (1.96 for α=0.05 two-tailed)
- `Z_β` = Z-score for power 1-β (0.84 for power=0.80)
- `p_pooled` = (p_control + p_treatment) / 2
- `δ` = p_treatment - p_control (absolute difference)

### 2.2 Parameters

**Inputs:**
- **Baseline conversion rate (p_control):** 3.0%
  - Source: Historical data (90-day average)
  - Justification: Stable metric, seasonal adjustments applied

- **Minimum Detectable Effect (MDE):** 5% relative
  - Absolute: 0.15 percentage points
  - Justification: Smallest business-meaningful improvement
  - Consideration: Balances statistical feasibility with business impact

- **Significance level (α):** 0.05
  - Standard: 95% confidence
  - One-sided vs two-sided: Two-sided for conservatism

- **Statistical power (1-β):** 80%
  - Standard: 80% probability of detecting true effect
  - Type II error: 20% chance of false negative

### 2.3 Sample Size Calculation

```python
from scipy import stats
import numpy as np

def calculate_sample_size(p_control, mde, alpha=0.05, power=0.80):
    """
    Calculate required sample size for two-sample proportion test.
    """
    # Convert relative to absolute MDE
    p_treatment = p_control * (1 + mde)
    
    # Pooled proportion
    p_pooled = (p_control + p_treatment) / 2
    
    # Standard error
    se = np.sqrt(2 * p_pooled * (1 - p_pooled))
    
    # Effect size
    delta = p_treatment - p_control
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = stats.norm.ppf(power)
    
    # Sample size per variant
    n = ((z_alpha + z_beta) * se / delta) ** 2
    
    return int(np.ceil(n))

n_per_variant = calculate_sample_size(
    p_control=0.03,
    mde=0.05,
    alpha=0.05,
    power=0.80
)

print(f"Required sample size: {n_per_variant:,} per variant")
# Output: Required sample size: 207,940 per variant
```

**Result:**
- Per variant: 207,940 users
- Total: 415,880 users
- At 11K DAU: ~19 days per variant (38 days total)

**Actual Experiment:**
- Per variant: 25,000 users
- Duration: 21 days
- **Note:** Under-powered for 5% MDE; can detect larger effects (~13% MDE at 80% power)

---

## 3. Validity Checks

### 3.1 Sample Ratio Mismatch (SRM)

**Purpose:** Verify randomization is working correctly

**Method:** Chi-square goodness-of-fit test

```python
from scipy.stats import chisquare

def check_srm(n_control, n_treatment, expected_ratio=0.5):
    """
    Test for sample ratio mismatch.
    """
    observed = np.array([n_control, n_treatment])
    total = n_control + n_treatment
    expected = np.array([total * expected_ratio, total * (1 - expected_ratio)])
    
    chi2, p_value = chisquare(observed, expected)
    
    # Very conservative threshold (p < 0.001)
    srm_detected = p_value < 0.001
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'ratio': n_treatment / n_control,
        'srm_detected': srm_detected
    }
```

**Decision Rule:**
- p < 0.001: SRM detected, investigate randomization
- p ≥ 0.001: No SRM, proceed with analysis

**Why 0.001?**
- More conservative than standard α=0.05
- Reduces false positives from natural variance
- Industry best practice (Kohavi et al., 2020)

### 3.2 Covariate Balance

**Purpose:** Verify randomization balanced observable characteristics

**Method:** Two-proportion z-test for each segment

```python
def check_covariate_balance(data, covariates):
    """
    Test balance of pre-treatment covariates.
    """
    results = []
    
    for covariate in covariates:
        # Proportion in control
        control = data[data['variant'] == 'control']
        p_control = (control[covariate] == 1).mean()
        n_control = len(control)
        
        # Proportion in treatment
        treatment = data[data['variant'] == 'treatment']
        p_treatment = (treatment[covariate] == 1).mean()
        n_treatment = len(treatment)
        
        # Z-test
        p_pooled = (p_control * n_control + p_treatment * n_treatment) / (n_control + n_treatment)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        z = (p_treatment - p_control) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        results.append({
            'covariate': covariate,
            'p_control': p_control,
            'p_treatment': p_treatment,
            'difference': p_treatment - p_control,
            'p_value': p_value,
            'balanced': p_value > 0.05
        })
    
    return pd.DataFrame(results)
```

**Covariates Checked:**
- User segment (new, casual, power)
- Device type
- Geographic region
- Historical conversion rate quartile

**Decision Rule:**
- All p-values > 0.05: Balanced, proceed
- Any p-value < 0.05: Imbalance detected, use covariate adjustment

---

## 4. Primary Analysis

### 4.1 Two-Sample Proportion Test

**Null Hypothesis:** H₀: p_treatment = p_control  
**Alternative Hypothesis:** H₁: p_treatment ≠ p_control (two-tailed)

**Test Statistic:**

```
Z = (p̂_treatment - p̂_control) / SE

where:
SE = sqrt(p̂_pooled * (1 - p̂_pooled) * (1/n_control + 1/n_treatment))
p̂_pooled = (x_control + x_treatment) / (n_control + n_treatment)
```

**Implementation:**

```python
def two_proportion_test(data, metric='converted', level='user'):
    """
    Perform two-sample proportion test.
    """
    # Aggregate to user level
    if level == 'user':
        user_data = data.groupby(['user_id', 'variant'])[metric].max().reset_index()
    else:
        user_data = data[['user_id', 'variant', metric]].copy()
    
    # Separate variants
    control = user_data[user_data['variant'] == 'control'][metric]
    treatment = user_data[user_data['variant'] == 'treatment'][metric]
    
    # Proportions
    n_control = len(control)
    n_treatment = len(treatment)
    p_control = control.mean()
    p_treatment = treatment.mean()
    
    # Pooled proportion
    x_control = control.sum()
    x_treatment = treatment.sum()
    p_pooled = (x_control + x_treatment) / (n_control + n_treatment)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    # Z-statistic
    z = (p_treatment - p_control) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Confidence interval for difference
    se_diff = np.sqrt(
        p_control * (1 - p_control) / n_control + 
        p_treatment * (1 - p_treatment) / n_treatment
    )
    ci_lower = (p_treatment - p_control) - 1.96 * se_diff
    ci_upper = (p_treatment - p_control) + 1.96 * se_diff
    
    # Relative lift
    relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
    
    return {
        'n_control': n_control,
        'n_treatment': n_treatment,
        'p_control': p_control,
        'p_treatment': p_treatment,
        'absolute_lift': p_treatment - p_control,
        'relative_lift': relative_lift,
        'relative_lift_pct': relative_lift * 100,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z_statistic': z,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 4.2 Confidence Intervals

**95% Confidence Interval for Difference:**

```
CI = (p̂_treatment - p̂_control) ± 1.96 * SE_diff

where:
SE_diff = sqrt(p̂_control * (1 - p̂_control) / n_control + 
               p̂_treatment * (1 - p̂_treatment) / n_treatment)
```

**95% CI for Relative Lift:**

```
CI_relative = CI_absolute / p̂_control
```

**Interpretation:**
- If CI excludes 0: Statistically significant
- If CI lower bound > MDE: Practically significant

### 4.3 Decision Criteria

**Statistical Significance:**
- p < 0.05 (95% confidence)
- CI does not include 0

**Practical Significance:**
- Relative lift ≥ 5%
- Lower bound of CI > 0%

**Combined Decision:**

| Statistical | Practical | Decision |
|-------------|-----------|----------|
| Yes | Yes | **SHIP** |
| Yes | No | Iterate (model not strong enough) |
| No | Yes | Run longer (underpowered) |
| No | No | Do not ship |

---

## 5. Segmentation Analysis

### 5.1 Purpose

Test if treatment effect varies by user segment:
- **New users:** 0-2 sessions
- **Casual users:** 3-10 sessions
- **Power users:** 11+ sessions

### 5.2 Method

Run separate two-proportion tests for each segment:

```python
def segment_analysis(data, segments, metric='converted'):
    """
    Analyze treatment effect by segment.
    """
    results = []
    
    for segment in segments:
        segment_data = data[data['segment'] == segment]
        result = two_proportion_test(segment_data, metric)
        result['segment'] = segment
        results.append(result)
    
    return pd.DataFrame(results)
```

### 5.3 Interpretation

**Heterogeneous Treatment Effects:**
- If segments differ significantly, personalize rollout
- Prioritize high-lift segments
- Investigate null/negative segments

**Example:**
- Power users: +15% (significant) → Launch first
- Casual users: +5% (significant) → Launch second
- New users: -2% (not significant) → Exclude or improve cold-start

### 5.4 Multiple Testing Correction

**Problem:** Testing 3 segments inflates Type I error

**Solution:** Bonferroni correction

```
α_adjusted = α / k = 0.05 / 3 = 0.0167
```

**Decision:** Segment result significant if p < 0.0167

---

## 6. Variance Reduction (CUPED)

### 6.1 Motivation

**Problem:** High variance reduces statistical power

**Solution:** CUPED (Controlled-experiment Using Pre-Experiment Data)
- Use pre-experiment metrics as covariates
- Reduce variance via regression adjustment
- Increase sensitivity without larger samples

### 6.2 Method

**Formula:**

```
Y_CUPED = Y - θ * (X - E[X])

where:
Y = Post-experiment metric
X = Pre-experiment covariate
θ = Cov(Y, X) / Var(X)
E[X] = Mean of X
```

**Implementation:**

```python
def cuped_adjustment(data, metric, covariate):
    """
    Apply CUPED variance reduction.
    """
    # Calculate theta
    cov = data[metric].cov(data[covariate])
    var = data[covariate].var()
    theta = cov / var if var > 0 else 0
    
    # Adjust metric
    covariate_mean = data[covariate].mean()
    data['metric_cuped'] = data[metric] - theta * (data[covariate] - covariate_mean)
    
    # Variance reduction
    var_original = data[metric].var()
    var_cuped = data['metric_cuped'].var()
    var_reduction = (var_original - var_cuped) / var_original
    
    return data, var_reduction
```

### 6.3 Benefits

**Variance Reduction:**
- Typically 10-40% reduction
- Higher correlation = more reduction
- Increases effective power

**No Bias:**
- CUPED is unbiased (Deng et al., 2013)
- Same point estimate, tighter CI
- More sensitive to true effects

---

## 7. Bayesian Analysis

### 7.1 Model

**Likelihood:** Binomial  
**Prior:** Beta(α, β) = Beta(1, 1) = Uniform(0, 1)  
**Posterior:** Beta(α + successes, β + failures)

### 7.2 Implementation

```python
def bayesian_ab_test(data, metric='converted', prior_alpha=1, prior_beta=1, n_samples=100000):
    """
    Bayesian A/B test with Beta-Binomial model.
    """
    # Aggregate to user level
    user_data = data.groupby(['user_id', 'variant'])[metric].max().reset_index()
    
    # Control
    control = user_data[user_data['variant'] == 'control'][metric]
    control_success = control.sum()
    control_n = len(control)
    control_failure = control_n - control_success
    
    # Treatment
    treatment = user_data[user_data['variant'] == 'treatment'][metric]
    treatment_success = treatment.sum()
    treatment_n = len(treatment)
    treatment_failure = treatment_n - treatment_success
    
    # Posteriors
    post_control_alpha = prior_alpha + control_success
    post_control_beta = prior_beta + control_failure
    
    post_treatment_alpha = prior_alpha + treatment_success
    post_treatment_beta = prior_beta + treatment_failure
    
    # Sample from posteriors
    control_samples = np.random.beta(post_control_alpha, post_control_beta, n_samples)
    treatment_samples = np.random.beta(post_treatment_alpha, post_treatment_beta, n_samples)
    
    # Probability treatment > control
    prob_treatment_better = (treatment_samples > control_samples).mean()
    
    # Credible intervals
    control_ci = np.percentile(control_samples, [2.5, 97.5])
    treatment_ci = np.percentile(treatment_samples, [2.5, 97.5])
    
    # Relative lift
    lift_samples = (treatment_samples - control_samples) / control_samples
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])
    
    # Expected loss
    expected_loss_treatment = (control_samples - treatment_samples).clip(min=0).mean()
    expected_loss_control = (treatment_samples - control_samples).clip(min=0).mean()
    
    return {
        'prob_treatment_better': prob_treatment_better,
        'control_ci': control_ci,
        'treatment_ci': treatment_ci,
        'lift_ci_pct': lift_ci * 100,
        'expected_loss_treatment': expected_loss_treatment,
        'expected_loss_control': expected_loss_control
    }
```

### 7.3 Decision Rule

**Bayesian Decision Thresholds:**
- P(treatment > control) > 0.95: **Launch treatment**
- P(treatment > control) < 0.05: **Keep control**
- 0.05 ≤ P(treatment > control) ≤ 0.95: **Inconclusive, run longer**

**Expected Loss:**
- Maximum acceptable loss: 0.1% absolute
- If expected_loss_treatment < 0.001: Safe to launch

---

## 8. Sensitivity Analysis

### 8.1 Purpose

Test robustness of results to analytical choices

### 8.2 Variations Tested

**1. Exclude Novelty Period**
- Remove first 2 days
- Test if effect stabilizes

**2. Outlier Removal**
- Exclude orders > $1000
- Test if extreme values drive results

**3. Different Significance Levels**
- α = 0.01 (more conservative)
- α = 0.10 (more liberal)

**4. Session-Level Analysis**
- Compare to user-level
- Assess repeated measures bias

### 8.3 Interpretation

**Robust Result:**
- Consistent across all sensitivity tests
- High confidence in findings

**Fragile Result:**
- Changes with analytical choices
- Interpret with caution

---

## 9. Multiple Testing

### 9.1 Problem

Testing multiple metrics increases Type I error (false positives)

**Family-Wise Error Rate (FWER):**
```
FWER = 1 - (1 - α)^k

For k=5 metrics at α=0.05:
FWER = 1 - (1 - 0.05)^5 = 0.226 (22.6%)
```

### 9.2 Solutions

**1. Pre-specify Primary Metric**
- Only one hypothesis test controls error
- Secondary metrics are exploratory

**2. Bonferroni Correction**
```
α_adjusted = α / k
```
- Conservative (may reduce power)
- Use when all metrics equally important

**3. False Discovery Rate (FDR)**
- Benjamini-Hochberg procedure
- Less conservative than Bonferroni
- Controls proportion of false positives

**Our Approach:**
- Primary metric (conversion): α = 0.05
- Secondary metrics: Bonferroni α = 0.05/3 = 0.0167
- Guardrails: Separate tests (not in family)

---

## 10. Assumptions and Limitations

### 10.1 Assumptions

**1. Independent and Identically Distributed (IID)**
- Users are independent
- Mitigation: User-level randomization

**2. Stable Unit Treatment Value Assumption (SUTVA)**
- No spillover between units
- One user's treatment doesn't affect others
- Limitation: Network effects (e.g., viral features) violate this

**3. Random Assignment**
- Users randomly assigned to variants
- Verification: SRM checks, balance checks

**4. No Post-Treatment Bias**
- No selection into outcome
- All users who click can convert

### 10.2 Limitations

**1. External Validity**
- Results apply to this platform, this time period
- May not generalize to other contexts

**2. Statistical Power**
- Underpowered for 5% MDE (only ~50% power)
- Can detect larger effects (~10-15%)

**3. Short Duration**
- 3 weeks may miss long-term effects
- Learning effects, novelty decay, seasonality

**4. Metric Limitations**
- Conversion rate is proxy for business value
- Doesn't capture customer lifetime value
- Doesn't account for margin differences

### 10.3 Threats to Validity

**Internal Validity:**
- ✅ Randomization: Verified via SRM
- ✅ Balance: Verified via covariate checks
- ⚠️ Attrition: Assume minimal (check dropout rates)

**External Validity:**
- ⚠️ Generalizability: Specific to current user base
- ⚠️ Seasonality: January may differ from other months

**Statistical Conclusion Validity:**
- ✅ Appropriate tests: Two-proportion test is correct
- ⚠️ Power: Limited for small effects
- ✅ Multiple testing: Controlled via Bonferroni

---

## References

1. Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.

2. Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data." *Proceedings of the Sixth ACM International Conference on Web Search and Data Mining*, 123-132.

3. Gelman, A., et al. (2013). *Bayesian Data Analysis*, Third Edition. CRC Press.

4. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

5. Netflix Tech Blog. (2016). "It's All A/Bout Testing: The Netflix Experimentation Platform." https://netflixtechblog.com/

6. Optimizely Stats Engine. (2015). "Stats Engine Technical Paper." https://www.optimizely.com/

---

**Document Control:**
- Version: 1.0
- Last Updated: January 2024
- Authors: Data Science Team
- Reviewers: Statistical Consulting, Product Analytics
