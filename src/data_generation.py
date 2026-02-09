"""
A/B Test Data Generation for E-Commerce Recommendation Engine
==============================================================

This module generates realistic A/B test data simulating an experiment where:
- Control: Rule-based recommendations
- Treatment: ML-powered personalized recommendations

Business Context:
- 50,000+ users over 3-4 weeks
- Multiple user segments with different behaviors
- Realistic temporal patterns (weekday/weekend effects)
- Treatment novelty effect (early spike, then stabilization)
- Guardrail metrics that might degrade slightly

Author: Data Science Team
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class ABTestDataGenerator:
    """
    Generates realistic A/B test data for e-commerce recommendations.
    
    Features:
    - User segmentation (new, casual, power users)
    - Temporal effects (weekend vs weekday)
    - Treatment novelty effect
    - Segment-specific treatment effects
    - Realistic noise and variance
    """
    
    def __init__(self, 
                 n_users: int = 50000,
                 n_weeks: int = 3,
                 random_seed: int = 42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        n_users : int
            Total number of users in the experiment
        n_weeks : int
            Duration of experiment in weeks
        random_seed : int
            Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_weeks = n_weeks
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Baseline metrics (Control group)
        self.baseline_conversion_rate = 0.03  # 3% baseline CR
        self.baseline_ctr = 0.15  # 15% click-through rate
        self.baseline_aov = 75.0  # $75 average order value
        self.baseline_page_load = 1.2  # 1.2 seconds
        self.baseline_engagement = 180  # 180 seconds
        
        # Treatment effects by segment
        self.treatment_effects = {
            'new': 0.03,      # 3% relative lift for new users
            'casual': 0.05,   # 5% relative lift for casual users  
            'power': 0.08     # 8% relative lift for power users
        }
        
    def generate_user_attributes(self) -> pd.DataFrame:
        """
        Generate user-level attributes including segment and variant assignment.
        
        Returns:
        --------
        pd.DataFrame with user_id, segment, variant
        """
        users = pd.DataFrame({
            'user_id': range(1, self.n_users + 1)
        })
        
        # User segmentation (realistic distribution)
        segment_probs = [0.30, 0.50, 0.20]  # new, casual, power
        users['segment'] = np.random.choice(
            ['new', 'casual', 'power'],
            size=self.n_users,
            p=segment_probs
        )
        
        # Stratified randomization (50-50 split within each segment)
        variants = []
        for segment in ['new', 'casual', 'power']:
            segment_users = users[users['segment'] == segment].index
            n_segment = len(segment_users)
            segment_variants = np.random.choice(
                ['control', 'treatment'],
                size=n_segment,
                p=[0.5, 0.5]
            )
            variants.extend(segment_variants)
        
        users['variant'] = variants
        
        return users
    
    def generate_temporal_pattern(self, week: int, day_of_week: int) -> float:
        """
        Generate temporal multiplier for activity levels.
        
        Parameters:
        -----------
        week : int
            Week number (0-indexed)
        day_of_week : int
            Day of week (0=Monday, 6=Sunday)
            
        Returns:
        --------
        float : activity multiplier
        """
        # Weekend effect (higher activity on Sat/Sun)
        weekend_boost = 1.3 if day_of_week >= 5 else 1.0
        
        # Novelty effect for treatment (decays over time)
        # Week 0: 1.15x, Week 1: 1.08x, Week 2+: 1.0x
        novelty_decay = max(1.0, 1.15 - (week * 0.07))
        
        return weekend_boost * novelty_decay
    
    def generate_sessions(self, users: pd.DataFrame) -> pd.DataFrame:
        """
        Generate session-level data with browsing behavior.
        
        Parameters:
        -----------
        users : pd.DataFrame
            User attributes dataframe
            
        Returns:
        --------
        pd.DataFrame with session-level metrics
        """
        sessions = []
        start_date = datetime(2024, 1, 1)
        
        for _, user in users.iterrows():
            user_id = user['user_id']
            segment = user['segment']
            variant = user['variant']
            
            # Session frequency by segment
            session_freq = {
                'new': np.random.randint(1, 4),      # 1-3 sessions
                'casual': np.random.randint(2, 8),   # 2-7 sessions
                'power': np.random.randint(5, 15)    # 5-14 sessions
            }
            
            n_sessions = session_freq[segment]
            
            for _ in range(n_sessions):
                # Random day in experiment period
                day_offset = np.random.randint(0, self.n_weeks * 7)
                session_date = start_date + timedelta(days=day_offset)
                week = day_offset // 7
                day_of_week = session_date.weekday()
                
                # Temporal multiplier
                temporal_mult = self.generate_temporal_pattern(week, day_of_week)
                
                # Base click probability (CTR)
                base_ctr = self.baseline_ctr
                
                # Treatment effect on CTR
                if variant == 'treatment':
                    treatment_lift = self.treatment_effects[segment]
                    ctr = base_ctr * (1 + treatment_lift) * temporal_mult
                else:
                    ctr = base_ctr * temporal_mult
                
                # Add noise
                ctr = min(0.95, max(0.05, ctr + np.random.normal(0, 0.02)))
                
                # Clicked on recommendations?
                clicked = np.random.random() < ctr
                
                # Conversion (only if clicked)
                converted = False
                revenue = 0.0
                
                if clicked:
                    # Base conversion rate (among clickers)
                    base_cr_given_click = self.baseline_conversion_rate / self.baseline_ctr
                    
                    if variant == 'treatment':
                        treatment_lift = self.treatment_effects[segment]
                        cr_given_click = base_cr_given_click * (1 + treatment_lift * 1.2)
                    else:
                        cr_given_click = base_cr_given_click
                    
                    # Add noise
                    cr_given_click = min(0.5, max(0.01, cr_given_click + np.random.normal(0, 0.03)))
                    
                    converted = np.random.random() < cr_given_click
                    
                    if converted:
                        # AOV varies by segment
                        aov_by_segment = {
                            'new': self.baseline_aov * 0.85,
                            'casual': self.baseline_aov,
                            'power': self.baseline_aov * 1.35
                        }
                        
                        mean_aov = aov_by_segment[segment]
                        
                        # Treatment slightly increases AOV
                        if variant == 'treatment':
                            mean_aov *= 1.03
                        
                        # Revenue with noise
                        revenue = max(10, np.random.gamma(
                            shape=4,
                            scale=mean_aov/4
                        ))
                
                # Page load time (guardrail metric)
                # Treatment increases load time slightly
                base_load = self.baseline_page_load
                if variant == 'treatment':
                    base_load *= 1.08  # 8% slower
                
                page_load_time = max(0.5, np.random.gamma(
                    shape=3,
                    scale=base_load/3
                ))
                
                # Engagement duration (guardrail metric)
                # Treatment might decrease engagement if recs are too good (faster purchase)
                base_engagement = self.baseline_engagement
                if variant == 'treatment' and converted:
                    base_engagement *= 0.92  # Faster path to purchase
                
                engagement_duration = max(10, np.random.gamma(
                    shape=2,
                    scale=base_engagement/2
                ))
                
                sessions.append({
                    'user_id': user_id,
                    'session_id': f"{user_id}_{len(sessions)}",
                    'session_date': session_date,
                    'week': week,
                    'day_of_week': day_of_week,
                    'segment': segment,
                    'variant': variant,
                    'clicked_recommendation': int(clicked),
                    'converted': int(converted),
                    'revenue': round(revenue, 2),
                    'page_load_time_sec': round(page_load_time, 2),
                    'engagement_duration_sec': round(engagement_duration, 1)
                })
        
        return pd.DataFrame(sessions)
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """
        Generate complete A/B test dataset.
        
        Returns:
        --------
        pd.DataFrame with all session-level data
        """
        print("Generating user attributes...")
        users = self.generate_user_attributes()
        
        print(f"Generating sessions for {self.n_users} users...")
        sessions = self.generate_sessions(users)
        
        print(f"Generated {len(sessions)} sessions")
        
        return sessions


def main():
    """Generate and save A/B test data."""
    
    print("=" * 70)
    print("A/B Test Data Generation")
    print("=" * 70)
    
    # Initialize generator
    generator = ABTestDataGenerator(
        n_users=50000,
        n_weeks=3,
        random_seed=42
    )
    
    # Generate data
    data = generator.generate_complete_dataset()
    
    # Save to CSV
    import os
    output_path = os.path.join(os.path.dirname(__file__), '../data/ab_test_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    
    print(f"\n✓ Data saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Sessions: {len(data):,}")
    print(f"Unique Users: {data['user_id'].nunique():,}")
    print(f"Date Range: {data['session_date'].min()} to {data['session_date'].max()}")
    
    print("\n--- Variant Distribution ---")
    print(data['variant'].value_counts())
    
    print("\n--- Segment Distribution ---")
    print(data['segment'].value_counts())
    
    print("\n--- Key Metrics by Variant ---")
    metrics_by_variant = data.groupby('variant').agg({
        'clicked_recommendation': 'mean',
        'converted': 'mean',
        'revenue': 'mean',
        'page_load_time_sec': 'mean',
        'engagement_duration_sec': 'mean'
    }).round(4)
    
    print(metrics_by_variant)
    
    print("\n--- User-Level Conversion Rate ---")
    user_conversions = data.groupby(['user_id', 'variant']).agg({
        'converted': 'max'  # User converted at least once
    }).reset_index()
    
    user_cr = user_conversions.groupby('variant')['converted'].mean()
    print(user_cr)
    
    print("\n✓ Data generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
