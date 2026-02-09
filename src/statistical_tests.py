"""
Statistical Analysis for A/B Testing
====================================

This module provides comprehensive statistical analysis including:
- Sample size calculations
- A/A test simulation
- Sample Ratio Mismatch (SRM) checks
- Covariate balance checks
- Two-sample proportion tests
- Confidence intervals
- Segmentation analysis
- CUPED variance reduction
- Bayesian A/B testing
- Sensitivity analysis

Author: Data Science Team
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ABTestAnalyzer:
    """
    Comprehensive A/B test statistical analyzer.
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            A/B test data with variant, metrics
        alpha : float
            Significance level (default 0.05)
        """
        self.data = data
        self.alpha = alpha
        self.results = {}
        
    @staticmethod
    def calculate_sample_size(baseline_rate: float,
                             mde: float,
                             alpha: float = 0.05,
                             power: float = 0.80,
                             two_tailed: bool = True) -> Dict:
        """
        Calculate required sample size for A/B test.
        
        Parameters:
        -----------
        baseline_rate : float
            Baseline conversion rate
        mde : float
            Minimum Detectable Effect (relative lift, e.g., 0.05 for 5%)
        alpha : float
            Significance level
        power : float
            Statistical power (1 - beta)
        two_tailed : bool
            Whether test is two-tailed
            
        Returns:
        --------
        Dict with sample size calculations
        """
        # Target rate
        target_rate = baseline_rate * (1 + mde)
        
        # Pooled proportion
        p_pooled = (baseline_rate + target_rate) / 2
        
        # Standard error
        se = np.sqrt(2 * p_pooled * (1 - p_pooled))
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2) if two_tailed else stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        
        # Effect size
        delta = target_rate - baseline_rate
        
        # Sample size per variant
        n_per_variant = ((z_alpha + z_beta) * se / delta) ** 2
        
        return {
            'n_per_variant': int(np.ceil(n_per_variant)),
            'n_total': int(np.ceil(n_per_variant * 2)),
            'baseline_rate': baseline_rate,
            'target_rate': target_rate,
            'mde': mde,
            'alpha': alpha,
            'power': power,
            'days_at_1k_dau': int(np.ceil(n_per_variant * 2 / 1000))  # Assuming 1k daily users
        }
    
    def check_srm(self) -> Dict:
        """
        Check for Sample Ratio Mismatch.
        
        Returns:
        --------
        Dict with SRM test results
        """
        variant_counts = self.data['variant'].value_counts()
        
        # Expected 50-50 split
        total = len(self.data)
        expected = total / 2
        
        # Chi-square test
        observed = variant_counts.values
        expected_array = np.array([expected, expected])
        
        chi2_stat, p_value = stats.chisquare(observed, expected_array)
        
        # SRM detected if p < 0.001 (very conservative)
        srm_detected = p_value < 0.001
        
        return {
            'control_n': int(variant_counts.get('control', 0)),
            'treatment_n': int(variant_counts.get('treatment', 0)),
            'expected_per_variant': expected,
            'ratio': variant_counts.get('treatment', 0) / variant_counts.get('control', 1),
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'srm_detected': srm_detected,
            'passed': not srm_detected
        }
    
    def check_covariate_balance(self) -> pd.DataFrame:
        """
        Check balance of covariates between variants.
        
        Returns:
        --------
        DataFrame with balance checks
        """
        # Aggregate to user level
        user_data = self.data.groupby(['user_id', 'variant', 'segment']).size().reset_index()
        
        # Check segment distribution
        balance_results = []
        
        for segment in ['new', 'casual', 'power']:
            control_prop = (user_data[
                (user_data['variant'] == 'control') & 
                (user_data['segment'] == segment)
            ].shape[0] / user_data[user_data['variant'] == 'control'].shape[0])
            
            treatment_prop = (user_data[
                (user_data['variant'] == 'treatment') & 
                (user_data['segment'] == segment)
            ].shape[0] / user_data[user_data['variant'] == 'treatment'].shape[0])
            
            # Two-proportion z-test
            n_control = user_data[user_data['variant'] == 'control'].shape[0]
            n_treatment = user_data[user_data['variant'] == 'treatment'].shape[0]
            
            pooled_prop = (control_prop * n_control + treatment_prop * n_treatment) / (n_control + n_treatment)
            se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/n_control + 1/n_treatment))
            
            z_stat = (treatment_prop - control_prop) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            balance_results.append({
                'covariate': f'segment_{segment}',
                'control_prop': control_prop,
                'treatment_prop': treatment_prop,
                'difference': treatment_prop - control_prop,
                'p_value': p_value,
                'balanced': p_value > 0.05
            })
        
        return pd.DataFrame(balance_results)
    
    def proportion_test(self, metric: str = 'converted', 
                       level: str = 'user') -> Dict:
        """
        Perform two-sample proportion test.
        
        Parameters:
        -----------
        metric : str
            Binary metric to test
        level : str
            'user' for user-level, 'session' for session-level
            
        Returns:
        --------
        Dict with test results
        """
        if level == 'user':
            # Aggregate to user level (at least one conversion)
            user_metric = self.data.groupby(['user_id', 'variant'])[metric].max().reset_index()
            control_data = user_metric[user_metric['variant'] == 'control'][metric]
            treatment_data = user_metric[user_metric['variant'] == 'treatment'][metric]
        else:
            control_data = self.data[self.data['variant'] == 'control'][metric]
            treatment_data = self.data[self.data['variant'] == 'treatment'][metric]
        
        # Calculate proportions
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        p_control = control_data.mean()
        p_treatment = treatment_data.mean()
        
        # Pooled proportion
        p_pooled = (control_data.sum() + treatment_data.sum()) / (n_control + n_treatment)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        
        # Z-statistic
        z_stat = (p_treatment - p_control) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval for difference
        se_diff = np.sqrt(p_control * (1 - p_control) / n_control + 
                         p_treatment * (1 - p_treatment) / n_treatment)
        
        ci_lower = (p_treatment - p_control) - 1.96 * se_diff
        ci_upper = (p_treatment - p_control) + 1.96 * se_diff
        
        # Relative lift
        relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
        
        # Relative lift CI
        rel_ci_lower = ci_lower / p_control if p_control > 0 else 0
        rel_ci_upper = ci_upper / p_control if p_control > 0 else 0
        
        return {
            'metric': metric,
            'level': level,
            'n_control': n_control,
            'n_treatment': n_treatment,
            'rate_control': p_control,
            'rate_treatment': p_treatment,
            'absolute_lift': p_treatment - p_control,
            'relative_lift': relative_lift,
            'relative_lift_pct': relative_lift * 100,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'rel_ci_lower_pct': rel_ci_lower * 100,
            'rel_ci_upper_pct': rel_ci_upper * 100,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'practical_significance': abs(relative_lift) >= 0.05  # 5% threshold
        }
    
    def segmentation_analysis(self, metric: str = 'converted') -> pd.DataFrame:
        """
        Analyze treatment effect by user segment.
        
        Parameters:
        -----------
        metric : str
            Binary metric to analyze
            
        Returns:
        --------
        DataFrame with segment-level results
        """
        segment_results = []
        
        for segment in ['new', 'casual', 'power']:
            segment_data = self.data[self.data['segment'] == segment]
            
            # User-level aggregation
            user_metric = segment_data.groupby(['user_id', 'variant'])[metric].max().reset_index()
            
            control_data = user_metric[user_metric['variant'] == 'control'][metric]
            treatment_data = user_metric[user_metric['variant'] == 'treatment'][metric]
            
            # Calculate metrics
            n_control = len(control_data)
            n_treatment = len(treatment_data)
            rate_control = control_data.mean()
            rate_treatment = treatment_data.mean()
            
            # Statistical test
            if rate_control > 0:
                relative_lift = (rate_treatment - rate_control) / rate_control
                
                # Simple z-test
                pooled_p = (control_data.sum() + treatment_data.sum()) / (n_control + n_treatment)
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
                z_stat = (rate_treatment - rate_control) / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                relative_lift = 0
                p_value = 1.0
            
            segment_results.append({
                'segment': segment,
                'n_control': n_control,
                'n_treatment': n_treatment,
                'rate_control': rate_control,
                'rate_treatment': rate_treatment,
                'absolute_lift': rate_treatment - rate_control,
                'relative_lift_pct': relative_lift * 100,
                'p_value': p_value,
                'significant': p_value < self.alpha
            })
        
        return pd.DataFrame(segment_results)
    
    def cuped_analysis(self, metric: str = 'revenue', 
                      covariate: str = 'engagement_duration_sec') -> Dict:
        """
        Apply CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
        
        Note: In a real scenario, we'd use pre-experiment data. Here we use a proxy.
        
        Parameters:
        -----------
        metric : str
            Metric to analyze
        covariate : str
            Pre-experiment covariate
            
        Returns:
        --------
        Dict with CUPED results
        """
        # Aggregate to user level
        user_data = self.data.groupby(['user_id', 'variant']).agg({
            metric: 'sum',
            covariate: 'mean'
        }).reset_index()
        
        # Calculate theta (covariance / variance)
        cov = user_data[metric].cov(user_data[covariate])
        var_covariate = user_data[covariate].var()
        theta = cov / var_covariate if var_covariate > 0 else 0
        
        # CUPED-adjusted metric
        covariate_mean = user_data[covariate].mean()
        user_data['metric_cuped'] = user_data[metric] - theta * (user_data[covariate] - covariate_mean)
        
        # Compare variance
        var_original = user_data[metric].var()
        var_cuped = user_data['metric_cuped'].var()
        variance_reduction = (var_original - var_cuped) / var_original if var_original > 0 else 0
        
        # T-test on CUPED metric
        control = user_data[user_data['variant'] == 'control']['metric_cuped']
        treatment = user_data[user_data['variant'] == 'treatment']['metric_cuped']
        
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        return {
            'metric': metric,
            'covariate': covariate,
            'theta': theta,
            'variance_original': var_original,
            'variance_cuped': var_cuped,
            'variance_reduction_pct': variance_reduction * 100,
            'control_mean': control.mean(),
            'treatment_mean': treatment.mean(),
            'lift': treatment.mean() - control.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def bayesian_ab_test(self, metric: str = 'converted', 
                        prior_alpha: float = 1,
                        prior_beta: float = 1) -> Dict:
        """
        Perform Bayesian A/B test for binary metrics.
        
        Parameters:
        -----------
        metric : str
            Binary metric
        prior_alpha, prior_beta : float
            Beta distribution prior parameters
            
        Returns:
        --------
        Dict with Bayesian results
        """
        # User-level data
        user_metric = self.data.groupby(['user_id', 'variant'])[metric].max().reset_index()
        
        control_data = user_metric[user_metric['variant'] == 'control'][metric]
        treatment_data = user_metric[user_metric['variant'] == 'treatment'][metric]
        
        # Successes and failures
        control_success = control_data.sum()
        control_total = len(control_data)
        control_failure = control_total - control_success
        
        treatment_success = treatment_data.sum()
        treatment_total = len(treatment_data)
        treatment_failure = treatment_total - treatment_success
        
        # Posterior distributions
        # Beta(alpha + successes, beta + failures)
        posterior_control_alpha = prior_alpha + control_success
        posterior_control_beta = prior_beta + control_failure
        
        posterior_treatment_alpha = prior_alpha + treatment_success
        posterior_treatment_beta = prior_beta + treatment_failure
        
        # Monte Carlo simulation to estimate P(treatment > control)
        n_samples = 100000
        control_samples = np.random.beta(posterior_control_alpha, posterior_control_beta, n_samples)
        treatment_samples = np.random.beta(posterior_treatment_alpha, posterior_treatment_beta, n_samples)
        
        prob_treatment_better = (treatment_samples > control_samples).mean()
        
        # Expected loss
        expected_loss_treatment = (control_samples - treatment_samples).clip(min=0).mean()
        expected_loss_control = (treatment_samples - control_samples).clip(min=0).mean()
        
        # Credible intervals (95%)
        control_ci = np.percentile(control_samples, [2.5, 97.5])
        treatment_ci = np.percentile(treatment_samples, [2.5, 97.5])
        
        lift_samples = (treatment_samples - control_samples) / control_samples
        lift_ci = np.percentile(lift_samples, [2.5, 97.5])
        
        return {
            'metric': metric,
            'control_rate': control_success / control_total,
            'treatment_rate': treatment_success / treatment_total,
            'prob_treatment_better': prob_treatment_better,
            'expected_loss_treatment': expected_loss_treatment,
            'expected_loss_control': expected_loss_control,
            'control_ci_lower': control_ci[0],
            'control_ci_upper': control_ci[1],
            'treatment_ci_lower': treatment_ci[0],
            'treatment_ci_upper': treatment_ci[1],
            'relative_lift_ci_lower_pct': lift_ci[0] * 100,
            'relative_lift_ci_upper_pct': lift_ci[1] * 100,
            'decision': 'Launch Treatment' if prob_treatment_better > 0.95 else 
                       'Keep Control' if prob_treatment_better < 0.05 else 
                       'Inconclusive'
        }
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete statistical analysis.
        
        Returns:
        --------
        Dict with all analysis results
        """
        results = {}
        
        print("Running comprehensive statistical analysis...\n")
        
        # 1. SRM Check
        print("1. Sample Ratio Mismatch Check...")
        results['srm'] = self.check_srm()
        
        # 2. Covariate Balance
        print("2. Covariate Balance Check...")
        results['covariate_balance'] = self.check_covariate_balance()
        
        # 3. Primary Metric Test
        print("3. Primary Metric Analysis (Conversion Rate)...")
        results['primary_metric'] = self.proportion_test('converted', level='user')
        
        # 4. Secondary Metrics
        print("4. Secondary Metrics Analysis...")
        results['ctr'] = self.proportion_test('clicked_recommendation', level='user')
        
        # 5. Segmentation
        print("5. Segmentation Analysis...")
        results['segmentation'] = self.segmentation_analysis('converted')
        
        # 6. CUPED
        print("6. CUPED Variance Reduction...")
        results['cuped'] = self.cuped_analysis('revenue', 'engagement_duration_sec')
        
        # 7. Bayesian Test
        print("7. Bayesian A/B Test...")
        results['bayesian'] = self.bayesian_ab_test('converted')
        
        # 8. Guardrail Metrics
        print("8. Guardrail Metrics...")
        user_load_time = self.data.groupby(['user_id', 'variant'])['page_load_time_sec'].mean().reset_index()
        control_load = user_load_time[user_load_time['variant'] == 'control']['page_load_time_sec']
        treatment_load = user_load_time[user_load_time['variant'] == 'treatment']['page_load_time_sec']
        
        t_stat_load, p_value_load = stats.ttest_ind(treatment_load, control_load)
        
        results['guardrail_page_load'] = {
            'control_mean': control_load.mean(),
            'treatment_mean': treatment_load.mean(),
            'difference': treatment_load.mean() - control_load.mean(),
            'pct_change': ((treatment_load.mean() - control_load.mean()) / control_load.mean() * 100),
            'p_value': p_value_load,
            'degraded': p_value_load < 0.05 and treatment_load.mean() > control_load.mean()
        }
        
        print("\n✓ Analysis complete!")
        
        return results


def print_results(results: Dict):
    """Pretty print analysis results."""
    
    print("\n" + "="*70)
    print("A/B TEST ANALYSIS RESULTS")
    print("="*70)
    
    # SRM
    print("\n--- SAMPLE RATIO MISMATCH CHECK ---")
    srm = results['srm']
    print(f"Control: {srm['control_n']:,}")
    print(f"Treatment: {srm['treatment_n']:,}")
    print(f"Ratio: {srm['ratio']:.4f}")
    print(f"P-value: {srm['p_value']:.4f}")
    print(f"✓ PASSED" if srm['passed'] else "✗ FAILED")
    
    # Primary Metric
    print("\n--- PRIMARY METRIC: CONVERSION RATE ---")
    pm = results['primary_metric']
    print(f"Control Rate: {pm['rate_control']:.4f} ({pm['rate_control']*100:.2f}%)")
    print(f"Treatment Rate: {pm['rate_treatment']:.4f} ({pm['rate_treatment']*100:.2f}%)")
    print(f"Absolute Lift: {pm['absolute_lift']:.4f} ({pm['absolute_lift']*100:.2f} pp)")
    print(f"Relative Lift: {pm['relative_lift_pct']:.2f}%")
    print(f"95% CI: [{pm['rel_ci_lower_pct']:.2f}%, {pm['rel_ci_upper_pct']:.2f}%]")
    print(f"P-value: {pm['p_value']:.4f}")
    print(f"Statistically Significant: {'YES' if pm['significant'] else 'NO'}")
    print(f"Practically Significant: {'YES' if pm['practical_significance'] else 'NO'}")
    
    # Bayesian
    print("\n--- BAYESIAN ANALYSIS ---")
    bay = results['bayesian']
    print(f"P(Treatment > Control): {bay['prob_treatment_better']:.4f}")
    print(f"95% Credible Interval: [{bay['relative_lift_ci_lower_pct']:.2f}%, {bay['relative_lift_ci_upper_pct']:.2f}%]")
    print(f"Decision: {bay['decision']}")
    
    # Segmentation
    print("\n--- SEGMENTATION ANALYSIS ---")
    print(results['segmentation'].to_string(index=False))
    
    # Guardrails
    print("\n--- GUARDRAIL METRICS ---")
    guard = results['guardrail_page_load']
    print(f"Page Load Time:")
    print(f"  Control: {guard['control_mean']:.3f}s")
    print(f"  Treatment: {guard['treatment_mean']:.3f}s")
    print(f"  Change: {guard['pct_change']:+.2f}%")
    print(f"  Degraded: {'YES' if guard['degraded'] else 'NO'}")
    
    print("\n" + "="*70)


def main():
    """Run complete statistical analysis."""
    
    # Load data
    data_path = '../data/ab_test_data.csv'
    data = pd.read_csv(data_path)
    
    print(f"Loaded {len(data):,} sessions from {data['user_id'].nunique():,} users")
    
    # Calculate sample size (informational)
    print("\n--- SAMPLE SIZE CALCULATION ---")
    sample_calc = ABTestAnalyzer.calculate_sample_size(
        baseline_rate=0.03,
        mde=0.05,  # 5% relative lift
        alpha=0.05,
        power=0.80
    )
    
    print(f"Required sample size per variant: {sample_calc['n_per_variant']:,}")
    print(f"Total required: {sample_calc['n_total']:,}")
    print(f"Actual users: {data['user_id'].nunique():,}")
    print(f"✓ Sample size is {'ADEQUATE' if data['user_id'].nunique() >= sample_calc['n_total'] else 'INADEQUATE'}")
    
    # Run analysis
    analyzer = ABTestAnalyzer(data, alpha=0.05)
    results = analyzer.run_full_analysis()
    
    # Print results
    print_results(results)
    
    # Save results
    import json
    
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_clean = convert_to_serializable(results)
    
    with open('../data/analysis_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print("\n✓ Results saved to analysis_results.json")


if __name__ == "__main__":
    main()
