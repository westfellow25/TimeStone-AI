"""
Monte Carlo Simulation Engine

Runs probabilistic simulations to predict transformation outcomes
with confidence intervals and risk analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation"""
    scenario_id: int
    scenario_name: str
    mean_roi: float
    median_roi: float
    std_dev: float
    confidence_90_lower: float
    confidence_90_upper: float
    success_probability: float
    risk_score: float
    iterations: int
    
    def __repr__(self):
        return (f"Scenario #{self.scenario_id}: {self.scenario_name}\n"
                f"  Mean ROI: {self.mean_roi:.1%}\n"
                f"  Success Probability: {self.success_probability:.1%}\n"
                f"  90% Confidence: [{self.confidence_90_lower:.1%}, "
                f"{self.confidence_90_upper:.1%}]")


class MonteCarloSimulator:
    """
    Monte Carlo simulation for transformation scenarios
    
    Simulates each scenario thousands of times with randomized
    parameters to understand probability distributions of outcomes.
    """
    
    def __init__(self, iterations: int = 1000, random_seed: int = 42):
        self.iterations = iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.results: List[SimulationResult] = []
    
    def simulate_scenario(
        self,
        scenario: Dict,
        baseline_revenue: float = 500_000_000
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a single scenario
        
        Args:
            scenario: Transformation scenario data
            baseline_revenue: Current company revenue
            
        Returns:
            Simulation results with statistics
        """
        revenue_impact = scenario["expected_impact"]["revenue_increase"]
        cost_impact = scenario["expected_impact"]["cost_reduction"]
        investment = scenario["investment_required"]
        impl_time = scenario["implementation_time_months"]
        risk_level = scenario["risk_level"]
        
        # Risk multipliers
        risk_multipliers = {
            "low": 1.1,
            "medium": 1.3,
            "high": 1.6
        }
        risk_factor = risk_multipliers.get(risk_level, 1.3)
        
        # Run Monte Carlo iterations
        roi_samples = []
        
        for _ in range(self.iterations):
            # Add uncertainty to projections
            actual_revenue_impact = np.random.normal(
                revenue_impact,
                revenue_impact * 0.3 * risk_factor  # Higher risk = more variance
            )
            
            actual_cost_impact = np.random.normal(
                cost_impact,
                cost_impact * 0.2 * risk_factor
            )
            
            # Implementation delay risk
            if risk_level == "high":
                delay_factor = np.random.uniform(1.0, 1.5)
                impl_time_actual = impl_time * delay_factor
            else:
                delay_factor = np.random.uniform(1.0, 1.2)
                impl_time_actual = impl_time * delay_factor
            
            # Cost overrun risk
            cost_overrun = np.random.uniform(1.0, 1.0 + (0.3 * risk_factor))
            actual_investment = investment * cost_overrun
            
            # Calculate ROI for this iteration
            annual_benefit = (
                baseline_revenue * actual_revenue_impact +
                baseline_revenue * 0.9 * actual_cost_impact  # 90% of cost reduction
            )
            
            # Simple ROI calculation (annualized)
            years_to_breakeven = impl_time_actual / 12
            total_benefit_3y = annual_benefit * 3  # 3-year horizon
            roi = (total_benefit_3y - actual_investment) / actual_investment
            
            roi_samples.append(roi)
        
        # Calculate statistics
        roi_array = np.array(roi_samples)
        mean_roi = np.mean(roi_array)
        median_roi = np.median(roi_array)
        std_dev = np.std(roi_array)
        
        # 90% confidence interval
        confidence_90_lower = np.percentile(roi_array, 5)
        confidence_90_upper = np.percentile(roi_array, 95)
        
        # Success probability (ROI > 0)
        success_count = np.sum(roi_array > 0)
        success_probability = success_count / self.iterations
        
        # Risk score (inverse of success probability, scaled)
        risk_score = 1.0 - success_probability
        
        return SimulationResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            mean_roi=mean_roi,
            median_roi=median_roi,
            std_dev=std_dev,
            confidence_90_lower=confidence_90_lower,
            confidence_90_upper=confidence_90_upper,
            success_probability=success_probability,
            risk_score=risk_score,
            iterations=self.iterations
        )
    
    def simulate_all_scenarios(
        self,
        scenarios: List[Dict],
        baseline_revenue: float = 500_000_000
    ) -> List[SimulationResult]:
        """
        Simulate all scenarios
        
        Args:
            scenarios: List of transformation scenarios
            baseline_revenue: Current company revenue
            
        Returns:
            List of simulation results
        """
        print(f"Running Monte Carlo simulation with {self.iterations} iterations per scenario...")
        print(f"Total scenarios: {len(scenarios)}")
        print("=" * 70)
        
        self.results = []
        
        for i, scenario in enumerate(scenarios, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(scenarios)} scenarios completed...")
            
            result = self.simulate_scenario(scenario, baseline_revenue)
            self.results.append(result)
        
        print(f"\n{'=' * 70}")
        print(f"Simulation complete! Analyzed {len(scenarios)} scenarios")
        
        return self.results
    
    def get_top_scenarios(
        self,
        n: int = 3,
        sort_by: str = "success_probability"
    ) -> List[SimulationResult]:
        """
        Get top N scenarios by specified metric
        
        Args:
            n: Number of scenarios to return
            sort_by: Metric to sort by ('success_probability', 'mean_roi', 'median_roi')
            
        Returns:
            Top N scenarios
        """
        if not self.results:
            return []
        
        sorted_results = sorted(
            self.results,
            key=lambda x: getattr(x, sort_by),
            reverse=True
        )
        
        return sorted_results[:n]
    
    def save_results(self, filepath: str):
        """Save simulation results to file"""
        data = {
            "simulation_parameters": {
                "iterations": self.iterations,
                "random_seed": self.random_seed,
                "total_scenarios": len(self.results)
            },
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "scenario_name": r.scenario_name,
                    "mean_roi": r.mean_roi,
                    "median_roi": r.median_roi,
                    "std_dev": r.std_dev,
                    "confidence_90_lower": r.confidence_90_lower,
                    "confidence_90_upper": r.confidence_90_upper,
                    "success_probability": r.success_probability,
                    "risk_score": r.risk_score
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Load scenarios
    print("Loading scenarios...")
    with open("ktz_scenarios.json", 'r') as f:
        data = json.load(f)
    
    scenarios = data["scenarios"]
    print(f"Loaded {len(scenarios)} scenarios for {data['company']}")
    print()
    
    # Create simulator
    simulator = MonteCarloSimulator(iterations=1000, random_seed=42)
    
    # Run simulations
    results = simulator.simulate_all_scenarios(
        scenarios,
        baseline_revenue=500_000_000  # KTZ baseline: $500M
    )
    
    # Get TOP-3 by success probability
    print("\n" + "=" * 70)
    print("TOP-3 SCENARIOS BY SUCCESS PROBABILITY")
    print("=" * 70)
    
    top_3 = simulator.get_top_scenarios(n=3, sort_by="success_probability")
    
    for i, result in enumerate(top_3, 1):
        print(f"\n🏆 RANK #{i}")
        print("-" * 70)
        print(result)
        print(f"  Risk Score: {result.risk_score:.3f}")
        print(f"  Standard Deviation: {result.std_dev:.3f}")
    
    # Save results
    simulator.save_results("ktz_simulation_results.json")
    print(f"\n{'=' * 70}")
    print("✅ Results saved to ktz_simulation_results.json")
