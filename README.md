# A/B Testing Project: E-Commerce Recommendation Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **[🔴 Live Streamlit Demo](https://optimizing-e-commerce-recommendations-using-a-b-testing-4yq3zq.streamlit.app/)**

> A complete, production-grade A/B testing project demonstrating rigorous statistical methodology, realistic data simulation, and executive-level reporting for optimizing e-commerce recommendations.

## 🎯 Project Overview
This project simulates and analyzes an A/B test comparing **ML-powered personalized recommendations** against **rule-based recommendations** in an e-commerce setting. It demonstrates end-to-end data science workflow from experimental design through statistical analysis to business recommendations.
Project Link : https://optimizing-e-commerce-recommendations-using-a-b-testing-4yq3zq.streamlit.app/

### Business Problem

An e-commerce platform wants to replace its rule-based recommendation engine with ML-powered personalized recommendations to improve:
- **Primary KPI:** Conversion Rate (target: +5% relative lift)
- **Secondary KPIs:** Click-Through Rate, Revenue Per User
- **Guardrails:** Page load time, user engagement

### Hypothesis

ML-powered personalized recommendations will increase conversion rate by at least 5% relative lift compared to rule-based recommendations, without degrading user experience.

## 📊 Key Results

| Metric | Control | Treatment | Lift | Significance |
|--------|---------|-----------|------|--------------|
| **Conversion Rate** | 15.12% | 16.40% | **+8.45%** | ✅ p<0.001 |
| Click-Through Rate | 17.70% | 18.61% | +5.14% | ✅ p<0.001 |
| Page Load Time | 1.216s | 1.310s | +7.70% | ⚠️ Degraded |

**Bayesian Analysis:** 99.99% probability that treatment is superior

**Recommendation:** ✅ **LAUNCH** with performance optimization

**Estimated Impact:** $XX,XXX additional annual revenue

## 🏗️ Project Structure

```
ab_testing_project/
├── src/
│   ├── data_generation.py      # Realistic A/B test data simulation
│   ├── statistical_tests.py    # Comprehensive statistical analysis
│   ├── visualization.py        # Executive-ready visualizations
│   └── app.py                  # Interactive Streamlit dashboard
├── data/
│   ├── ab_test_data.csv       # Generated dataset (237K sessions, 50K users)
│   └── analysis_results.json  # Statistical test results
├── figures/                    # Generated visualizations
│   ├── metric_comparison.png
│   ├── confidence_intervals.png
│   ├── time_series.png
│   ├── segment_heatmap.png
│   ├── guardrail_distribution.png
│   └── cumulative_conversions.png
├── docs/
│   ├── experiment_plan.md           # Experimental design document
│   ├── statistical_methodology.md   # Statistical approach documentation
│   └── final_report.md              # Executive summary report
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ab-testing-project.git
cd ab-testing-project

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# 1. Generate synthetic A/B test data
cd src
python data_generation.py

# 2. Run statistical analysis
python statistical_tests.py

# 3. Generate visualizations
python visualization.py

# 4. Launch interactive dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📁 Components

### 1. Data Generation (`data_generation.py`)

Simulates realistic A/B test data with:
- ✅ 50,000 users over 3 weeks
- ✅ User segmentation (new, casual, power users)
- ✅ Temporal patterns (weekend effects, novelty decay)
- ✅ Segment-specific treatment effects
- ✅ Realistic noise and variance
- ✅ Guardrail metric trade-offs

**Output:** `ab_test_data.csv` (237,319 sessions)

### 2. Statistical Analysis (`statistical_tests.py`)

Comprehensive testing framework:
- ✅ Sample size calculation (power analysis)
- ✅ Sample Ratio Mismatch (SRM) check
- ✅ Covariate balance verification
- ✅ Two-sample proportion tests with confidence intervals
- ✅ Segmentation analysis
- ✅ CUPED variance reduction
- ✅ Bayesian A/B testing
- ✅ Guardrail metric evaluation

**Output:** `analysis_results.json`

### 3. Visualizations (`visualization.py`)

Executive-ready charts:
- ✅ Metric comparison plots
- ✅ Confidence interval visualizations
- ✅ Time series analysis
- ✅ Segment performance heatmaps
- ✅ Guardrail metric distributions
- ✅ Cumulative conversion curves

**Output:** PNG files in `figures/`

### 4. Interactive Dashboard (`app.py`)

Multi-page Streamlit application:
- 📊 **Executive Summary:** Go/No-Go recommendation, ROI estimate
- 📈 **Statistical Analysis:** Detailed test results, SRM checks, Bayesian analysis
- 🎯 **Segment Analysis:** Treatment effects by user cohort
- 🚧 **Guardrail Metrics:** Performance impact assessment
- 📊 **Visualizations:** Interactive charts and graphs

## 📈 Methodology

### Experimental Design

- **Randomization:** Stratified by user segment (50-50 split)
- **Sample Size:** 50,000 users (powered for 5% MDE at 80% power)
- **Duration:** 3 weeks (21 days)
- **Analysis Level:** User-level (to account for repeat users)

### Statistical Approach

**Frequentist:**
- Two-sample proportion test for conversion rate
- 95% confidence intervals
- α = 0.05 significance threshold
- Bonferroni correction for multiple comparisons

**Bayesian:**
- Beta-Binomial conjugate priors
- 95% credible intervals
- Decision threshold: P(treatment > control) > 0.95

**Variance Reduction:**
- CUPED methodology using pre-experiment covariates
- ~XX% variance reduction achieved

### Validity Checks

✅ Sample Ratio Mismatch: p=0.584 (PASSED)
✅ Covariate Balance: All segments balanced
✅ Temporal Stability: Effects consistent over time

## 🎯 Business Impact

### Recommended Action

**✅ LAUNCH with Performance Optimization**

The treatment demonstrates:
- Strong statistical evidence (p<0.001)
- Practical significance (8.45% > 5% MDE)
- High Bayesian confidence (99.99%)

However, page load time degradation requires attention.

### Revenue Impact

Assuming:
- Daily active users: ~11,000
- Annual users: ~4,000,000
- Average order value: $75

**Projected Impact:**
- Additional conversions: ~51,200/year
- Revenue increase: ~$3,840,000/year

### Rollout Strategy

1. **Phase 1 (Month 1):** Launch to Power users only (+13.8% lift, proven segment)
2. **Phase 2 (Month 2):** Expand to Casual users (+8.9% lift, large segment)
3. **Phase 3 (Month 3):** Full rollout after performance optimization
4. **Optimization:** Parallel workstream to reduce page load time by 50%

## 🔬 Key Insights

### Segment Analysis

| Segment | Lift | Significance | Interpretation |
|---------|------|--------------|----------------|
| Power   | +13.8% | ✅ Significant | **Best performing** - prioritize this segment |
| Casual  | +8.9% | ✅ Significant | Large volume, strong lift |
| New     | -1.0% | ❌ Not Significant | No effect, needs investigation |

**Insight:** Power users benefit most from personalization, likely due to richer interaction history.

### Trade-Offs

✅ **Pros:**
- Significant conversion lift
- Revenue positive
- Works well for 70% of user base

⚠️ **Cons:**
- +7.7% page load time increase
- No benefit for new users
- Requires ML infrastructure

## 🔄 Next Steps

### Technical Optimizations

1. **Performance:**
   - Implement server-side caching (target: -50% load time)
   - Pre-compute recommendations for returning users
   - Optimize ML model inference

2. **New User Experience:**
   - Investigate why new users don't benefit
   - Develop cold-start algorithms
   - A/B test hybrid approach (rules + ML)

3. **Monitoring:**
   - Real-time performance dashboards
   - Automated alerting on metric degradation
   - Continuous learning pipeline

### Future Experiments

1. **Recommendation Display:** Test different layouts, counts, positions
2. **Personalization Depth:** Test varying levels of personalization
3. **Multi-Armed Bandits:** Dynamic allocation to best-performing variants
4. **Long-term Effects:** 6-month cohort analysis for customer lifetime value

## 📚 References

### Statistical Methods
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"

### Tools & Libraries
- Python 3.8+
- pandas, numpy, scipy
- matplotlib, seaborn
- streamlit (dashboard)

## 👨‍💻 About This Project
https://optimizing-e-commerce-recommendations-using-a-b-testing-4yq3zq.streamlit.app/

This portfolio project demonstrates:
- ✅ End-to-end A/B testing methodology
- ✅ Production-quality code with documentation
- ✅ Statistical rigor (frequentist + Bayesian)
- ✅ Business-focused insights
- ✅ Executive communication
- ✅ Realistic data simulation

**Applicable to:**
- E-commerce platforms (Amazon, eBay, Shopify)
- Content recommendations (Netflix, YouTube, Spotify)
- Product features (Microsoft 365, Xbox, Azure)
- Marketing campaigns (Bing Ads, LinkedIn)

## 📧 Contact

**Your Name**
- Email: charankarthiknayakanti@gmail.com
- LinkedIn: https://www.linkedin.com/in/charankarthiknayakanti/
- Portfolio: https://www.datascienceportfol.io/CHARANKARTHIKN?preview=True

## 📄 License

MIT License - see LICENSE file for details

---

**Note:** This is a synthetic demonstration project for portfolio purposes. All data is simulated and does not represent any real company.
