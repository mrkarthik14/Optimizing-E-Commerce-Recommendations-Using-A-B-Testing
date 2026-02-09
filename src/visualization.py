"""
A/B Test Visualization Module
==============================

This module creates executive-ready visualizations for A/B test results.

Includes:
- Metric comparison plots
- Confidence interval visualizations
- Time series analysis
- Segment heatmaps
- Guardrail metric distributions

Author: Data Science Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ABTestVisualizer:
    """
    Create publication-quality visualizations for A/B test results using matplotlib.
    """
    
    def __init__(self, data: pd.DataFrame, results: Dict):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            A/B test data
        results : Dict
            Statistical analysis results
        """
        self.data = data
        self.results = results
        self.colors = {
            'control': '#3B82F6',  # Blue
            'treatment': '#10B981'  # Green
        }
        
    def plot_metric_comparison(self, save_path: str = None):
        """
        Create bar chart comparing key metrics between variants.
        """
        # Prepare data
        metrics = ['clicked_recommendation', 'converted']
        metric_names = ['Click-Through Rate', 'Conversion Rate']
        
        user_data = self.data.groupby(['user_id', 'variant']).agg({
            'clicked_recommendation': 'max',
            'converted': 'max'
        }).reset_index()
        
        control_rates = []
        treatment_rates = []
        errors = []
        
        for metric in metrics:
            control = user_data[user_data['variant'] == 'control'][metric].mean()
            treatment = user_data[user_data['variant'] == 'treatment'][metric].mean()
            
            # Calculate 95% CI
            n_t = len(user_data[user_data['variant'] == 'treatment'])
            se = np.sqrt(treatment * (1-treatment) / n_t)
            
            control_rates.append(control * 100)
            treatment_rates.append(treatment * 100)
            errors.append(1.96 * se * 100)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, control_rates, width, label='Control',
                      color=self.colors['control'], alpha=0.8)
        bars2 = ax.bar(x + width/2, treatment_rates, width, label='Treatment',
                      color=self.colors['treatment'], alpha=0.8, yerr=errors, capsize=5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, control_rates)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        
        for i, (bar, val) in enumerate(zip(bars2, treatment_rates)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[i] + 0.5,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Key Metrics: Control vs Treatment', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confidence_intervals(self, save_path: str = None):
        """
        Plot confidence intervals for relative lift.
        """
        pm = self.results['primary_metric']
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot CI
        ax.plot([pm['rel_ci_lower_pct'], pm['rel_ci_upper_pct']], [0, 0],
               linewidth=3, color=self.colors['treatment'], label='95% CI')
        
        # Point estimate
        ax.scatter([pm['relative_lift_pct']], [0], s=150, color=self.colors['treatment'],
                  zorder=5, label='Point Estimate')
        
        # Reference lines
        ax.axvline(x=5, linestyle='--', color='red', alpha=0.7, label='MDE (5%)')
        ax.axvline(x=0, linestyle=':', color='gray', alpha=0.5)
        
        ax.set_xlabel('Relative Lift (%)', fontsize=12, fontweight='bold')
        ax.set_title('Treatment Effect: Relative Lift with 95% Confidence Interval',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.legend(loc='upper right')
        ax.grid(axis='x', alpha=0.3)
        
        # Add annotation
        ax.text(pm['relative_lift_pct'], 0.15, 
               f"{pm['relative_lift_pct']:.2f}%\n[{pm['rel_ci_lower_pct']:.2f}%, {pm['rel_ci_upper_pct']:.2f}%]",
               ha='center', va='bottom', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_time_series(self, save_path: str = None):
        """
        Plot conversion rate over time.
        """
        # Daily conversion rates
        daily_data = self.data.groupby(['session_date', 'variant', 'user_id']).agg({
            'converted': 'max'
        }).reset_index()
        
        daily_rates = daily_data.groupby(['session_date', 'variant']).agg({
            'converted': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for variant in ['control', 'treatment']:
            variant_data = daily_rates[daily_rates['variant'] == variant]
            ax.plot(pd.to_datetime(variant_data['session_date']), 
                   variant_data['converted'] * 100,
                   marker='o', linewidth=2, label=variant.capitalize(),
                   color=self.colors[variant], markersize=6)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Conversion Rate Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_segment_heatmap(self, save_path: str = None):
        """
        Create bar plot of treatment effects by segment.
        """
        seg_results = pd.DataFrame(self.results['segmentation'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create colors based on significance
        colors = []
        for _, row in seg_results.iterrows():
            if row['significant'] and row['relative_lift_pct'] > 0:
                colors.append('#10B981')  # Green
            elif row['significant'] and row['relative_lift_pct'] < 0:
                colors.append('#EF4444')  # Red
            else:
                colors.append('#9CA3AF')  # Gray
        
        bars = ax.bar(seg_results['segment'].str.capitalize(), 
                     seg_results['relative_lift_pct'],
                     color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, val, sig) in enumerate(zip(bars, seg_results['relative_lift_pct'], 
                                                  seg_results['significant'])):
            label = f"{val:.1f}%{'*' if sig else ''}"
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + (0.5 if val > 0 else -0.5),
                   label, ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=11, fontweight='bold')
        
        ax.axhline(y=0, color='gray', linewidth=1)
        ax.axhline(y=5, linestyle='--', color='red', alpha=0.7, label='MDE (5%)')
        
        ax.set_xlabel('User Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Lift (%)', fontsize=12, fontweight='bold')
        ax.set_title('Treatment Effect by User Segment (* = statistically significant)',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_guardrail_distribution(self, save_path: str = None):
        """
        Plot distribution of guardrail metric (page load time).
        """
        user_load = self.data.groupby(['user_id', 'variant']).agg({
            'page_load_time_sec': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for variant in ['control', 'treatment']:
            variant_data = user_load[user_load['variant'] == variant]['page_load_time_sec']
            ax.hist(variant_data, bins=50, alpha=0.6, label=variant.capitalize(),
                   color=self.colors[variant], edgecolor='black', linewidth=0.5)
        
        # Add mean lines
        control_mean = user_load[user_load['variant'] == 'control']['page_load_time_sec'].mean()
        treatment_mean = user_load[user_load['variant'] == 'treatment']['page_load_time_sec'].mean()
        
        ax.axvline(control_mean, color=self.colors['control'], linestyle='--', 
                  linewidth=2, label=f'Control Mean: {control_mean:.2f}s')
        ax.axvline(treatment_mean, color=self.colors['treatment'], linestyle='--',
                  linewidth=2, label=f'Treatment Mean: {treatment_mean:.2f}s')
        
        ax.set_xlabel('Page Load Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Page Load Time Distribution (Guardrail Metric)',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_cumulative_conversions(self, save_path: str = None):
        """
        Plot cumulative conversions over time.
        """
        # User first conversion
        user_conv = self.data[self.data['converted'] == 1].groupby(['user_id', 'variant']).agg({
            'session_date': 'min'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for variant in ['control', 'treatment']:
            variant_data = user_conv[user_conv['variant'] == variant].sort_values('session_date')
            variant_data['cumulative'] = range(1, len(variant_data) + 1)
            
            ax.plot(pd.to_datetime(variant_data['session_date']), 
                   variant_data['cumulative'],
                   linewidth=2, label=variant.capitalize(),
                   color=self.colors[variant])
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Conversions', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Conversions Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_all_visualizations(self, output_dir: str = '../figures'):
        """
        Generate and save all visualizations.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save figures
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualizations...")
        
        print("1. Metric comparison...")
        self.plot_metric_comparison(f'{output_dir}/metric_comparison.png')
        
        print("2. Confidence intervals...")
        self.plot_confidence_intervals(f'{output_dir}/confidence_intervals.png')
        
        print("3. Time series...")
        self.plot_time_series(f'{output_dir}/time_series.png')
        
        print("4. Segment analysis...")
        self.plot_segment_heatmap(f'{output_dir}/segment_heatmap.png')
        
        print("5. Guardrail metrics...")
        self.plot_guardrail_distribution(f'{output_dir}/guardrail_distribution.png')
        
        print("6. Cumulative conversions...")
        self.plot_cumulative_conversions(f'{output_dir}/cumulative_conversions.png')
        
        print(f"\n✓ All visualizations saved to {output_dir}/")



def main():
    """Generate all visualizations."""
    
    # Load data
    data = pd.read_csv('../data/ab_test_data.csv')
    data['session_date'] = pd.to_datetime(data['session_date'])
    
    # Load results
    with open('../data/analysis_results.json', 'r') as f:
        results = json.load(f)
    
    # Create visualizer
    viz = ABTestVisualizer(data, results)
    
    # Generate all plots
    viz.create_all_visualizations()
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
