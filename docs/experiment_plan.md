# Experimental Design Plan
## E-Commerce Recommendation Engine A/B Test

**Document Version:** 1.0  
**Date:** January 1, 2024  
**Status:** Approved  
**Author:** Data Science Team

---

## 1. Executive Summary

This document outlines the experimental design for testing ML-powered personalized recommendations against the current rule-based recommendation system.

**Objective:** Determine if ML-powered recommendations increase conversion rate by ≥5% relative lift without degrading user experience.

**Timeline:** 3 weeks (Jan 1 - Jan 21, 2024)  
**Sample Size:** 50,000 users  
**Decision Criteria:** Statistical significance (p<0.05) AND practical significance (≥5% lift)

---

## 2. Business Context

### 2.1 Problem Statement

Our current rule-based recommendation system shows recommendations based on:
- Product category matching
- Basic popularity rankings
- Simple business rules (e.g., margin, inventory)

We hypothesize that personalized ML recommendations will better match user preferences and drive higher conversion.

### 2.2 Success Criteria

**Primary Goal:**
- Increase conversion rate by ≥5% relative lift

**Secondary Goals:**
- Maintain or improve click-through rate
- Increase average order value
- Maintain or improve revenue per user

**Guardrails:**
- Page load time increase <10%
- No significant decrease in engagement duration
- No increase in bounce rate

### 2.3 Business Impact

**Assumptions:**
- Current conversion rate: 3%
- Average order value: $75
- Daily active users: ~11,000
- Annual revenue from recommendations: ~$9M

**Projected Impact (5% lift):**
- Additional daily conversions: ~16
- Additional annual revenue: ~$450K
- ROI on ML infrastructure: 3-5x within first year

---

## 3. Hypothesis

**Null Hypothesis (H₀):**  
ML recommendations have no effect on conversion rate: CR_treatment = CR_control

**Alternative Hypothesis (H₁):**  
ML recommendations increase conversion rate: CR_treatment > CR_control

**Statistical Framework:**
- One-sided test (directional hypothesis)
- Significance level: α = 0.05
- Statistical power: 1-β = 0.80

---

## 4. Experimental Design

### 4.1 Randomization

**Unit of Randomization:** User (user_id)

**Allocation:** 50-50 split
- Control: 50% of users see rule-based recommendations
- Treatment: 50% of users see ML-powered recommendations

**Stratification:** By user segment
- New users (0-2 sessions): 30%
- Casual users (3-10 sessions): 50%
- Power users (11+ sessions): 20%

**Why user-level?**
- Prevents user confusion from seeing different recommendations
- Accounts for learning effects
- More conservative than session-level analysis

### 4.2 Sample Size Calculation

**Inputs:**
- Baseline conversion rate: p₁ = 0.03
- Minimum Detectable Effect (MDE): 5% relative = 0.0015 absolute
- Target conversion rate: p₂ = 0.0315
- Significance level: α = 0.05
- Power: 1-β = 0.80

**Formula:**
```
n = 2 * ((Z_α + Z_β)² * p_pooled * (1 - p_pooled)) / δ²

where:
- Z_α = 1.96 (for α=0.05, two-tailed)
- Z_β = 0.84 (for power=0.80)
- p_pooled = (p₁ + p₂) / 2 = 0.03075
- δ = p₂ - p₁ = 0.0015
```

**Required Sample Size:**
- Per variant: 207,940 users
- Total: 415,880 users

**Actual Sample:**
- Per variant: 25,000 users
- Total: 50,000 users

**Note:** This is under-powered for a 5% MDE. To achieve adequate power, we would need:
- 8-12 weeks at current DAU (~11,000/day)
- OR higher MDE (e.g., 10% requires ~52,000 total)

For this demonstration, we proceed with awareness that statistical power is limited.

### 4.3 Duration

**Test Duration:** 3 weeks (21 days)

**Rationale:**
- Captures full weekly cycle (multiple weekends)
- Allows novelty effect to stabilize
- Long enough for users to have multiple sessions
- Short enough for timely business decision

**Weekly Patterns Considered:**
- Weekend traffic typically 30% higher
- Friday-Sunday conversions 20% higher
- Need full weeks to avoid weekly seasonality bias

### 4.4 Variant Descriptions

**Control (50%):**
- Current rule-based recommendation engine
- Shows products based on:
  - Category matching (40% weight)
  - Popularity ranking (30% weight)
  - Margin optimization (20% weight)
  - Inventory availability (10% weight)

**Treatment (50%):**
- ML-powered personalized recommendations
- Features:
  - Collaborative filtering (user-user similarity)
  - Item-based recommendations
  - User browsing history
  - Purchase history
  - Real-time behavior signals
- Model: Matrix Factorization with deep learning enhancements
- Inference: Real-time (<100ms p99)

---

## 5. Metrics

### 5.1 Primary Metric

**Conversion Rate (User-Level)**

**Definition:**  
Proportion of users who made at least one purchase during the experiment period

**Calculation:**
```
CR = (# users with ≥1 conversion) / (# total users)
```

**Why user-level?**
- Avoids repeated measures bias
- More conservative than session-level
- Aligns with business objective (acquire converting users)

**Analysis Method:**
- Two-sample proportion test
- 95% confidence intervals
- Both absolute and relative lift

### 5.2 Secondary Metrics

**Click-Through Rate (CTR)**
- Definition: % of users who clicked ≥1 recommendation
- Indicates engagement with recommendation module

**Average Order Value (AOV)**
- Definition: Mean revenue per converted user
- Tests if personalization affects basket size

**Revenue Per User (RPU)**
- Definition: Total revenue / total users
- Combines conversion and AOV effects

### 5.3 Guardrail Metrics

**Page Load Time**
- Threshold: <10% increase
- Why: ML inference may slow page rendering
- Measurement: p50, p90, p99 latencies

**Engagement Duration**
- Threshold: No significant decrease
- Why: Ensure users aren't frustrated
- Measurement: Mean session duration

**Bounce Rate**
- Threshold: No significant increase
- Why: Monitor if recommendations are relevant

---

## 6. Implementation Details

### 6.1 Randomization Implementation

```python
def assign_variant(user_id, segment):
    """
    Deterministic assignment based on user_id hash.
    Ensures consistent experience across sessions.
    """
    hash_val = hashlib.md5(f"{user_id}_{segment}".encode()).hexdigest()
    hash_int = int(hash_val, 16)
    
    if hash_int % 2 == 0:
        return 'control'
    else:
        return 'treatment'
```

### 6.2 Data Collection

**Events Tracked:**
- User ID, variant assignment, segment
- Session start/end timestamps
- Recommendation impressions
- Recommendation clicks
- Purchase events (product_id, revenue, timestamp)
- Page load metrics (DOM ready, full load)
- Engagement metrics (scroll depth, time on page)

**Data Storage:**
- Real-time event stream (Kafka/Kinesis)
- Aggregated to data warehouse (Snowflake/BigQuery)
- Daily batch for analysis

### 6.3 Quality Checks

**Pre-Launch:**
- [ ] A/A test (both variants see control for 2 days)
- [ ] Sample Ratio Mismatch monitoring
- [ ] Covariate balance check
- [ ] Instrumentation validation

**During Test:**
- [ ] Daily SRM checks
- [ ] Metric monitoring dashboards
- [ ] Anomaly detection alerts
- [ ] Data quality checks

**Post-Test:**
- [ ] Statistical validity checks
- [ ] Segment analysis
- [ ] Guardrail metric review
- [ ] Decision documentation

---

## 7. Analysis Plan

### 7.1 Statistical Tests

**Primary Analysis:**
1. Two-sample proportion test (conversion rate)
2. 95% confidence interval for difference
3. Relative lift calculation
4. P-value for significance

**Secondary Analyses:**
1. Segment-specific treatment effects
2. Time-to-convert analysis
3. CUPED variance reduction
4. Bayesian posterior probability

**Sensitivity Analyses:**
1. Exclude first 2 days (novelty period)
2. Exclude outliers (>$1000 orders)
3. Different significance thresholds
4. Bonferroni correction for multiple tests

### 7.2 Decision Framework

**Ship Decision:**

| Criteria | Threshold | Status |
|----------|-----------|--------|
| Statistical Significance | p < 0.05 | ? |
| Practical Significance | Lift ≥ 5% | ? |
| Guardrail: Page Load | Increase < 10% | ? |
| Guardrail: Engagement | No decrease | ? |

**Decision Matrix:**
- All criteria met → **SHIP**
- Stat sig but not practical → **Iterate on model**
- Practical but not stat sig → **Run longer**
- Guardrails violated → **Optimize performance**
- No evidence of lift → **Do not ship**

### 7.3 Reporting

**Deliverables:**
1. Executive summary (1-pager)
2. Detailed statistical report
3. Interactive dashboard (Streamlit)
4. Segment deep-dive
5. Recommendations for next steps

**Stakeholders:**
- Executive leadership (go/no-go decision)
- Product managers (feature roadmap)
- Engineering (implementation)
- Data science (methodology)

---

## 8. Risks and Mitigation

### 8.1 Statistical Risks

**Risk:** Sample size inadequate for 5% MDE  
**Mitigation:** 
- Extend test duration if needed
- Consider higher MDE (10%) as acceptable
- Use Bayesian framework for early stopping

**Risk:** Sample ratio mismatch  
**Mitigation:**
- Automated daily monitoring
- Kill switch if SRM detected (p < 0.001)
- Root cause analysis of randomization

**Risk:** Multiple testing inflation  
**Mitigation:**
- Pre-specify all analyses
- Bonferroni correction for family-wise error
- Focus on single primary metric

### 8.2 Implementation Risks

**Risk:** ML service latency/failures  
**Mitigation:**
- Fallback to rule-based on timeout (>500ms)
- Extensive load testing pre-launch
- Gradual rollout (1% → 10% → 50%)

**Risk:** Data pipeline delays  
**Mitigation:**
- Real-time event validation
- Duplicate data warehouse writes
- Manual backup data pulls

**Risk:** Inconsistent user experience  
**Mitigation:**
- Sticky assignment via user_id
- Assignment stored in user profile
- QA testing across devices

### 8.3 Business Risks

**Risk:** Negative user feedback  
**Mitigation:**
- Qualitative user research (surveys, interviews)
- Monitor customer support tickets
- Quick rollback capability

**Risk:** Cannibalization of other revenue  
**Mitigation:**
- Track total site revenue, not just recommendation revenue
- Monitor non-recommended product sales
- Basket composition analysis

---

## 9. Timeline

| Week | Dates | Activities |
|------|-------|-----------|
| -2 | Dec 18-24 | Design finalization, code complete |
| -1 | Dec 25-31 | A/A test, QA, stakeholder review |
| 1 | Jan 1-7 | Launch 50-50 test, daily monitoring |
| 2 | Jan 8-14 | Continue test, interim analysis |
| 3 | Jan 15-21 | Complete test, final analysis |
| 4 | Jan 22-28 | Results presentation, decision |
| 5+ | Jan 29+ | Implementation/iteration based on decision |

---

## 10. Success Scenarios

### Scenario A: Significant Lift, No Guardrail Issues
**Outcome:** Full launch to 100%  
**Timeline:** 1 week gradual rollout  
**Next:** Optimize model, expand to other modules

### Scenario B: Significant Lift, Guardrail Issues
**Outcome:** Limited launch to power users only  
**Timeline:** 2 weeks optimization, then expand  
**Next:** Performance engineering workstream

### Scenario C: Positive but Not Significant
**Outcome:** Extend test 2-4 weeks  
**Timeline:** Re-evaluate after extended period  
**Next:** Consider Bayesian early stopping

### Scenario D: No Effect or Negative
**Outcome:** Do not launch  
**Timeline:** Immediate  
**Next:** Root cause analysis, model improvements

---

## 11. Approvals

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Data Science Lead | [Name] | [Date] | [Approved] |
| Product Manager | [Name] | [Date] | [Approved] |
| Engineering Lead | [Name] | [Date] | [Approved] |
| VP Product | [Name] | [Date] | [Approved] |

---

## 12. Appendices

### Appendix A: Metric Definitions (Full)

**Conversion Rate:**
- Numerator: COUNT(DISTINCT user_id WHERE converted = TRUE)
- Denominator: COUNT(DISTINCT user_id)
- Aggregation: User-level
- Window: Full experiment period

### Appendix B: Experiment Configuration

```yaml
experiment:
  name: "ml_recommendations_v1"
  start_date: "2024-01-01"
  end_date: "2024-01-21"
  
  variants:
    control:
      allocation: 0.5
      description: "Rule-based recommendations"
    treatment:
      allocation: 0.5
      description: "ML-powered recommendations"
  
  stratification:
    - segment
  
  metrics:
    primary:
      - conversion_rate
    secondary:
      - click_through_rate
      - average_order_value
      - revenue_per_user
    guardrails:
      - page_load_time
      - engagement_duration
      - bounce_rate
```

### Appendix C: References

1. Kohavi, R., et al. (2020). *Trustworthy Online Controlled Experiments*
2. Deng, A., et al. (2013). "Improving Sensitivity with Pre-Experiment Data" (CUPED)
3. Company A/B Testing Playbook (Internal)
4. Statistical Power Calculator: [link]

---

**Document Control:**
- Last Updated: January 1, 2024
- Version: 1.0
- Next Review: After experiment completion
