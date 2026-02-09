"""
A/B Test Results Dashboard
==========================

Interactive Streamlit dashboard for exploring A/B test results.

To run:
    streamlit run app.py

Author: Data Science Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="A/B Test Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .recommendation-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .go {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .no-go {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load A/B test data and results."""
    data = pd.read_csv('../data/ab_test_data.csv')
    data['session_date'] = pd.to_datetime(data['session_date'])
    
    with open('../data/analysis_results.json', 'r') as f:
        results = json.load(f)
    
    return data, results


def show_executive_summary(results):
    """Display executive summary page."""
    st.title("🎯 Executive Summary")
    st.markdown("---")
    
    # Get key metrics
    pm = results['primary_metric']
    bay = results['bayesian']
    
    # Decision recommendation
    if pm['significant'] and pm['practical_significance'] and pm['relative_lift'] > 0:
        decision = "✅ GO - Launch Treatment"
        decision_class = "go"
        explanation = """
        **Recommendation: LAUNCH THE TREATMENT**
        
        The ML-powered personalized recommendations demonstrate a statistically significant 
        and practically meaningful improvement over the rule-based system. The treatment 
        meets all success criteria with high confidence.
        """
    elif pm['significant'] and pm['relative_lift'] < 0:
        decision = "❌ NO-GO - Keep Control"
        decision_class = "no-go"
        explanation = """
        **Recommendation: DO NOT LAUNCH**
        
        The treatment shows a statistically significant negative impact. 
        We recommend keeping the current rule-based system.
        """
    else:
        decision = "⚠️ INCONCLUSIVE - Run Longer"
        decision_class = "no-go"
        explanation = """
        **Recommendation: EXTEND TEST DURATION**
        
        While directionally positive, the results do not yet meet statistical 
        significance thresholds. Recommend running the test longer.
        """
    
    st.markdown(f'<div class="recommendation-box {decision_class}"><h2>{decision}</h2>{explanation}</div>', 
                unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Conversion Rate Lift",
            f"{pm['relative_lift_pct']:.2f}%",
            f"{pm['absolute_lift']*100:.2f} pp",
            help="Relative improvement in conversion rate"
        )
    
    with col2:
        st.metric(
            "Statistical Confidence",
            f"{(1-pm['p_value'])*100:.1f}%",
            "Significant" if pm['significant'] else "Not Significant",
            help="Confidence that the effect is real"
        )
    
    with col3:
        st.metric(
            "Bayesian Probability",
            f"{bay['prob_treatment_better']*100:.1f}%",
            "Treatment Wins",
            help="Probability that treatment is better"
        )
    
    with col4:
        # Calculate estimated revenue impact
        control_n = pm['n_control']
        treatment_n = pm['n_treatment']
        avg_users_per_day = (control_n + treatment_n) / 21  # 3 weeks
        annual_users = avg_users_per_day * 365
        
        # Assuming average order value
        avg_order_value = 75  # From data generation
        
        # Additional conversions per year
        additional_conversions = annual_users * pm['absolute_lift']
        revenue_impact = additional_conversions * avg_order_value
        
        st.metric(
            "Estimated Annual Impact",
            f"${revenue_impact:,.0f}",
            f"+{additional_conversions:,.0f} conversions/year",
            help="Projected annual revenue increase"
        )
    
    # Detailed Results
    st.markdown("### 📈 Detailed Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Control Group**")
        st.write(f"- **Users:** {pm['n_control']:,}")
        st.write(f"- **Conversion Rate:** {pm['rate_control']*100:.2f}%")
        
    with col2:
        st.markdown("**Treatment Group**")
        st.write(f"- **Users:** {pm['n_treatment']:,}")
        st.write(f"- **Conversion Rate:** {pm['rate_treatment']*100:.2f}%")
    
    st.markdown("**Effect Size**")
    st.write(f"- **Absolute Lift:** {pm['absolute_lift']*100:.2f} percentage points")
    st.write(f"- **Relative Lift:** {pm['relative_lift_pct']:.2f}%")
    st.write(f"- **95% Confidence Interval:** [{pm['rel_ci_lower_pct']:.2f}%, {pm['rel_ci_upper_pct']:.2f}%]")
    st.write(f"- **P-value:** {pm['p_value']:.4f}")


def show_statistical_analysis(results):
    """Display statistical analysis page."""
    st.title("📊 Statistical Deep Dive")
    st.markdown("---")
    
    # Sample Ratio Mismatch
    st.markdown("### 🔍 Sample Ratio Mismatch (SRM) Check")
    srm = results['srm']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Control Group", f"{srm['control_n']:,}")
    with col2:
        st.metric("Treatment Group", f"{srm['treatment_n']:,}")
    with col3:
        status = "✅ PASSED" if srm['passed'] else "❌ FAILED"
        st.metric("SRM Status", status)
    
    st.write(f"**Ratio:** {srm['ratio']:.4f} (Expected: 1.000)")
    st.write(f"**P-value:** {srm['p_value']:.4f}")
    
    if srm['passed']:
        st.success("✓ No sample ratio mismatch detected. Randomization appears to be working correctly.")
    else:
        st.error("⚠️ Sample ratio mismatch detected! This may indicate issues with the randomization process.")
    
    # Covariate Balance
    st.markdown("### ⚖️ Covariate Balance")
    cov_balance = pd.DataFrame(results['covariate_balance'])
    
    st.dataframe(cov_balance.style.format({
        'control_prop': '{:.4f}',
        'treatment_prop': '{:.4f}',
        'difference': '{:.4f}',
        'p_value': '{:.4f}'
    }), use_container_width=True)
    
    if all(cov_balance['balanced']):
        st.success("✓ All covariates are balanced between control and treatment groups.")
    else:
        st.warning("⚠️ Some covariates show imbalance. This may affect the validity of results.")
    
    # Primary Metric Analysis
    st.markdown("### 🎯 Primary Metric: Conversion Rate")
    
    pm = results['primary_metric']
    
    st.markdown("**Two-Sample Proportion Test**")
    metrics_df = pd.DataFrame({
        'Metric': ['Sample Size', 'Conversion Rate', 'Absolute Lift', 'Relative Lift', 'Z-Statistic', 'P-Value'],
        'Control': [
            f"{pm['n_control']:,}",
            f"{pm['rate_control']*100:.2f}%",
            "-",
            "-",
            "-",
            "-"
        ],
        'Treatment': [
            f"{pm['n_treatment']:,}",
            f"{pm['rate_treatment']*100:.2f}%",
            f"{pm['absolute_lift']*100:.2f} pp",
            f"{pm['relative_lift_pct']:.2f}%",
            f"{pm['z_statistic']:.4f}",
            f"{pm['p_value']:.4f}"
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown(f"**95% Confidence Interval:** [{pm['rel_ci_lower_pct']:.2f}%, {pm['rel_ci_upper_pct']:.2f}%]")
    
    if pm['significant']:
        st.success(f"✓ The result is statistically significant at α=0.05 (p={pm['p_value']:.4f})")
    else:
        st.info(f"The result is not statistically significant at α=0.05 (p={pm['p_value']:.4f})")
    
    if pm['practical_significance']:
        st.success("✓ The effect exceeds the 5% practical significance threshold")
    else:
        st.warning("⚠️ The effect does not meet the 5% practical significance threshold")
    
    # Bayesian Analysis
    st.markdown("### 🎲 Bayesian Analysis")
    
    bay = results['bayesian']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "P(Treatment > Control)",
            f"{bay['prob_treatment_better']*100:.2f}%",
            help="Posterior probability that treatment is better"
        )
    
    with col2:
        st.metric(
            "Decision",
            bay['decision'],
            help="Bayesian decision based on 95% threshold"
        )
    
    st.markdown(f"**95% Credible Interval:** [{bay['relative_lift_ci_lower_pct']:.2f}%, {bay['relative_lift_ci_upper_pct']:.2f}%]")
    
    st.info("""
    **Interpretation:** A Bayesian probability >95% is generally considered strong evidence 
    for launching the treatment. Values between 5-95% suggest more data is needed.
    """)


def show_segment_analysis(data, results):
    """Display segmentation analysis page."""
    st.title("🎯 Segment Analysis")
    st.markdown("---")
    
    st.markdown("""
    This analysis shows how different user segments respond to the treatment.
    Understanding segment-specific effects helps target the rollout strategy.
    """)
    
    # Segment results
    seg_results = pd.DataFrame(results['segmentation'])
    
    st.markdown("### 📊 Treatment Effect by Segment")
    
    # Format the dataframe
    seg_display = seg_results.copy()
    seg_display['segment'] = seg_display['segment'].str.capitalize()
    seg_display['rate_control'] = (seg_display['rate_control'] * 100).round(2).astype(str) + '%'
    seg_display['rate_treatment'] = (seg_display['rate_treatment'] * 100).round(2).astype(str) + '%'
    seg_display['absolute_lift'] = (seg_display['absolute_lift'] * 100).round(2).astype(str) + ' pp'
    seg_display['relative_lift_pct'] = seg_display['relative_lift_pct'].round(2).astype(str) + '%'
    seg_display['p_value'] = seg_display['p_value'].round(4)
    seg_display['significant'] = seg_display['significant'].apply(lambda x: '✓' if x else '✗')
    
    seg_display = seg_display.rename(columns={
        'segment': 'Segment',
        'n_control': 'Control N',
        'n_treatment': 'Treatment N',
        'rate_control': 'Control CR',
        'rate_treatment': 'Treatment CR',
        'absolute_lift': 'Absolute Lift',
        'relative_lift_pct': 'Relative Lift',
        'p_value': 'P-Value',
        'significant': 'Significant'
    })
    
    st.dataframe(seg_display, use_container_width=True, hide_index=True)
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    # Find best and worst performing segments
    best_segment = seg_results.loc[seg_results['relative_lift_pct'].idxmax()]
    worst_segment = seg_results.loc[seg_results['relative_lift_pct'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best Performing Segment**")
        st.write(f"- **Segment:** {best_segment['segment'].capitalize()}")
        st.write(f"- **Lift:** {best_segment['relative_lift_pct']:.2f}%")
        st.write(f"- **Significant:** {'Yes' if best_segment['significant'] else 'No'}")
        
    with col2:
        st.markdown("**Lowest Performing Segment**")
        st.write(f"- **Segment:** {worst_segment['segment'].capitalize()}")
        st.write(f"- **Lift:** {worst_segment['relative_lift_pct']:.2f}%")
        st.write(f"- **Significant:** {'Yes' if worst_segment['significant'] else 'No'}")
    
    # Recommendations
    st.markdown("### 🎯 Recommendations")
    
    if best_segment['relative_lift_pct'] > 10 and best_segment['significant']:
        st.success(f"""
        **Prioritize {best_segment['segment'].capitalize()} Users:** 
        This segment shows exceptional lift ({best_segment['relative_lift_pct']:.1f}%). 
        Consider targeting ML recommendations to this group first for maximum impact.
        """)
    
    if worst_segment['relative_lift_pct'] < 0:
        st.warning(f"""
        **Monitor {worst_segment['segment'].capitalize()} Users:** 
        This segment shows negative lift. Consider excluding them from the rollout 
        or investigating why the treatment performs poorly for this group.
        """)


def show_guardrail_metrics(results):
    """Display guardrail metrics page."""
    st.title("🚧 Guardrail Metrics")
    st.markdown("---")
    
    st.markdown("""
    Guardrail metrics ensure that improvements in primary metrics don't come at 
    the cost of user experience or system performance.
    """)
    
    # Page Load Time
    st.markdown("### ⏱️ Page Load Time")
    
    guard = results['guardrail_page_load']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Control Mean", f"{guard['control_mean']:.3f}s")
    
    with col2:
        st.metric(
            "Treatment Mean", 
            f"{guard['treatment_mean']:.3f}s",
            f"{guard['pct_change']:+.2f}%"
        )
    
    with col3:
        status = "❌ Degraded" if guard['degraded'] else "✅ OK"
        st.metric("Status", status)
    
    st.markdown(f"**Difference:** {guard['difference']:.3f}s ({guard['pct_change']:+.2f}%)")
    st.markdown(f"**P-value:** {guard['p_value']:.4f}")
    
    if guard['degraded']:
        st.error("""
        ⚠️ **Page load time has significantly increased**
        
        The treatment adds ~8% to page load time. While conversion rate improved, 
        this degradation in performance needs attention. Consider:
        - Optimizing the ML model inference
        - Implementing caching strategies
        - Lazy-loading recommendations
        """)
    else:
        st.success("✓ Page load time is within acceptable limits")
    
    # Recommendations
    st.markdown("### 🎯 Next Steps")
    
    st.info("""
    **Performance Optimization Recommendations:**
    1. Profile the recommendation engine to identify bottlenecks
    2. Implement server-side caching for frequently accessed recommendations
    3. Consider pre-computing recommendations during off-peak hours
    4. A/B test lighter ML models with similar accuracy
    5. Monitor real user experience metrics (e.g., bounce rate, time on page)
    """)


def show_visualizations():
    """Display visualizations page."""
    st.title("📈 Visualizations")
    st.markdown("---")
    
    figures_dir = Path('../figures')
    
    if not figures_dir.exists():
        st.error("Figures directory not found. Please run visualization.py first.")
        return
    
    # Get all PNG files
    figures = list(figures_dir.glob('*.png'))
    
    if not figures:
        st.warning("No visualizations found. Please run visualization.py first.")
        return
    
    # Display each figure
    for fig_path in sorted(figures):
        st.markdown(f"### {fig_path.stem.replace('_', ' ').title()}")
        st.image(str(fig_path), use_container_width=True)
        st.markdown("---")


def main():
    """Main dashboard application."""
    
    # Sidebar
    st.sidebar.title("📊 A/B Test Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Executive Summary", 
         "📊 Statistical Analysis", 
         "🎯 Segment Analysis",
         "🚧 Guardrail Metrics",
         "📈 Visualizations"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Test")
    st.sidebar.info("""
    **Experiment:** ML Recommendations vs Rule-Based
    
    **Duration:** 3 weeks (Jan 1-21, 2024)
    
    **Sample Size:** 50,000 users
    
    **Primary Metric:** Conversion Rate
    
    **Hypothesis:** ML recommendations will increase CR by ≥5%
    """)
    
    # Load data
    try:
        data, results = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Display selected page
    if page == "🎯 Executive Summary":
        show_executive_summary(results)
    elif page == "📊 Statistical Analysis":
        show_statistical_analysis(results)
    elif page == "🎯 Segment Analysis":
        show_segment_analysis(data, results)
    elif page == "🚧 Guardrail Metrics":
        show_guardrail_metrics(results)
    elif page == "📈 Visualizations":
        show_visualizations()


if __name__ == "__main__":
    main()
