"""
Business impact simulation.
Phase 7: Compute revenue uplift from model improvements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class BusinessImpactSimulator:
    """Simulate business impact of improved recommendations."""
    
    def __init__(
        self,
        baseline_attach_rate: float,
        transformer_attach_rate: float,
        avg_addon_value: float = 60.0,
        monthly_orders: int = 10_000_000
    ):
        """
        Args:
            baseline_attach_rate: Current attach rate (0-1)
            transformer_attach_rate: Model attach rate (0-1)
            avg_addon_value: Average add-on value in ₹
            monthly_orders: Monthly order volume
        """
        self.baseline_attach_rate = baseline_attach_rate
        self.transformer_attach_rate = transformer_attach_rate
        self.avg_addon_value = avg_addon_value
        self.monthly_orders = monthly_orders
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute impact metrics."""
        
        baseline_addons = self.monthly_orders * self.baseline_attach_rate
        transformer_addons = self.monthly_orders * self.transformer_attach_rate
        incremental_addons = transformer_addons - baseline_addons
        
        baseline_revenue = baseline_addons * self.avg_addon_value
        transformer_revenue = transformer_addons * self.avg_addon_value
        incremental_revenue = incremental_addons * self.avg_addon_value
        
        attach_rate_uplift = (
            (self.transformer_attach_rate - self.baseline_attach_rate) /
            (self.baseline_attach_rate + 1e-8) * 100
        )
        
        return {
            'baseline_attach_rate': self.baseline_attach_rate,
            'transformer_attach_rate': self.transformer_attach_rate,
            'attach_rate_uplift_pct': attach_rate_uplift,
            'baseline_monthly_addons': baseline_addons,
            'transformer_monthly_addons': transformer_addons,
            'incremental_monthly_addons': incremental_addons,
            'baseline_monthly_revenue': baseline_revenue,
            'transformer_monthly_revenue': transformer_revenue,
            'incremental_monthly_revenue': incremental_revenue,
            'incremental_annual_revenue': incremental_revenue * 12,
        }
    
    def print_report(self):
        """Print business impact report."""
        metrics = self.compute_metrics()
        
        print("\n" + "="*70)
        print("BUSINESS IMPACT SIMULATION")
        print("="*70)
        
        print(f"\nAssumptions:")
        print(f"  Monthly Order Volume: {self.monthly_orders:,}")
        print(f"  Average Add-On Value: ₹{self.avg_addon_value:.2f}")
        
        print(f"\nCurrent Attach Rates:")
        print(f"  Baseline: {metrics['baseline_attach_rate']:.2%}")
        print(f"  Transformer: {metrics['transformer_attach_rate']:.2%}")
        print(f"  Uplift: {metrics['attach_rate_uplift_pct']:.2f}%")
        
        print(f"\nMonthly Metrics:")
        print(f"  Baseline Add-Ons: {metrics['baseline_monthly_addons']:,.0f}")
        print(f"  Transformer Add-Ons: {metrics['transformer_monthly_addons']:,.0f}")
        print(f"  Incremental Add-Ons: {metrics['incremental_monthly_addons']:,.0f}")
        
        print(f"\nRevenue Impact (Monthly):")
        print(f"  Baseline Revenue: ₹{metrics['baseline_monthly_revenue']:,.0f}")
        print(f"  Transformer Revenue: ₹{metrics['transformer_monthly_revenue']:,.0f}")
        print(f"  Incremental Revenue: ₹{metrics['incremental_monthly_revenue']:,.0f}")
        
        print(f"\nRevenue Impact (Annual):")
        print(f"  Incremental Annual Revenue: ₹{metrics['incremental_annual_revenue']:,.0f}")
        
        print()


def estimate_attach_rates_from_metrics(
    test_df: pd.DataFrame,
    baseline_precision_at_8: float,
    transformer_precision_at_8: float,
    current_attach_rate: float = 0.15
) -> Tuple[float, float]:
    """
    Estimate attach rates from model metrics.
    
    Rough heuristic:
    - Baseline precision@8 correlates to current attach rate
    - Improved precision translates to higher attach rate
    
    Returns:
        (estimated_baseline_attach_rate, estimated_transformer_attach_rate)
    """
    
    # Conservative estimate: precision@8 as a proxy for improvement
    # Baseline attach rate typically lower than precision (model only ranks top 8)
    # We'll use precision as relative improvement
    
    baseline_estimated = current_attach_rate * (1 + baseline_precision_at_8 * 0.1)
    transformer_estimated = current_attach_rate * (1 + transformer_precision_at_8 * 0.1)
    
    return baseline_estimated, transformer_estimated


def run_business_impact_analysis(
    baseline_precision_at_8: float,
    transformer_precision_at_8: float,
    current_attach_rate: float = 0.15,
    avg_addon_value: float = 60.0,
    monthly_orders: int = 10_000_000
) -> Dict:
    """
    Run complete business impact analysis.
    
    Returns:
        Dictionary with all impact metrics
    """
    
    # Estimate attach rates
    baseline_attach, transformer_attach = estimate_attach_rates_from_metrics(
        None,
        baseline_precision_at_8,
        transformer_precision_at_8,
        current_attach_rate
    )
    
    print("\n" + "="*70)
    print("ESTIMATED ATTACH RATES FROM MODEL METRICS")
    print("="*70)
    print(f"Current/Baseline: {baseline_attach:.2%}")
    print(f"With Transformer: {transformer_attach:.2%}")
    print()
    
    # Run simulator
    simulator = BusinessImpactSimulator(
        baseline_attach,
        transformer_attach,
        avg_addon_value,
        monthly_orders
    )
    
    simulator.print_report()
    
    return simulator.compute_metrics()
