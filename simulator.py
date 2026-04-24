"""
Clinical trial readout simulator.

Monte Carlo simulation of event-driven clinical trials to predict
when statistical milestones (interim and final analyses) will occur.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TrialAssumptions:
    """Container for all trial design inputs."""
    # Enrollment
    enrollment_start: datetime
    total_sample_size: int
    monthly_accrual_rate: float       # patients per month at steady state
    accrual_rampup_months: float      # linear ramp-up from 0 to steady state
    randomization_ratio: float        # treatment:control, e.g., 2.0 means 2:1

    # Survival assumptions
    control_median_months: float      # median PFS or OS in control arm
    target_hazard_ratio: float        # assumed treatment effect (HR < 1 favors treatment)
    dropout_rate_annual: float        # annual probability of dropout/censoring

    # Analysis plan
    alpha_two_sided: float            # typically 0.05
    power: float                      # typically 0.80 or 0.90
    information_fractions: list       # e.g., [0.5, 0.8, 1.0] for IA1, IA2, final


def required_events(hr: float, alpha: float, power: float, ratio: float) -> int:
    """
    Schoenfeld's formula: number of events needed for target power.

    This is a standard result in survival analysis. The intuition:
    smaller HR effects (HR closer to 1) need more events to detect,
    which is why trials in indications with weak expected effects
    need huge sample sizes and long follow-up.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    # allocation proportions
    p_treat = ratio / (1 + ratio)
    p_ctrl = 1 / (1 + ratio)
    numerator = (z_alpha + z_beta) ** 2
    denominator = p_treat * p_ctrl * (np.log(hr)) ** 2
    return int(np.ceil(numerator / denominator))


def simulate_enrollment_dates(assumptions: TrialAssumptions, rng: np.random.Generator) -> np.ndarray:
    """
    Generate enrollment dates for each patient.

    Models a linear ramp-up (sites activating gradually) followed by
    steady-state accrual. Returns array of months-from-trial-start
    for each enrolled patient.
    """
    n = assumptions.total_sample_size
    rate = assumptions.monthly_accrual_rate
    rampup = assumptions.accrual_rampup_months

    # Patients enrolled during ramp-up: area under the triangle
    rampup_patients = int(0.5 * rate * rampup)

    if rampup_patients >= n:
        # All enrollment happens during ramp-up (small trial / slow accrual)
        # Inverse CDF sampling from triangular distribution
        u = rng.uniform(0, 1, n)
        return rampup * np.sqrt(u)

    # Ramp-up phase: triangular distribution of enrollment times
    u = rng.uniform(0, 1, rampup_patients)
    rampup_times = rampup * np.sqrt(u)

    # Steady-state phase: uniform accrual after ramp-up
    remaining = n - rampup_patients
    steady_duration = remaining / rate
    steady_times = rng.uniform(rampup, rampup + steady_duration, remaining)

    return np.concatenate([rampup_times, steady_times])


def simulate_event_times(assumptions: TrialAssumptions, rng: np.random.Generator) -> tuple:
    """
    For each patient, simulate time-to-event and time-to-dropout.

    Uses exponential distribution (constant hazard). Could upgrade to
    Weibull later if non-proportional hazards become a concern.

    Returns: (event_times, is_treatment_arm) both in months from enrollment.
    """
    n = assumptions.total_sample_size
    ratio = assumptions.randomization_ratio

    # Assign arms based on randomization ratio
    p_treat = ratio / (1 + ratio)
    is_treatment = rng.uniform(0, 1, n) < p_treat

    # Control arm: hazard derived from median survival
    # For exponential: median = ln(2) / hazard
    control_hazard_monthly = np.log(2) / assumptions.control_median_months
    # Treatment arm hazard = control hazard * HR
    treatment_hazard_monthly = control_hazard_monthly * assumptions.target_hazard_ratio

    # Draw event times from exponential distribution
    hazards = np.where(is_treatment, treatment_hazard_monthly, control_hazard_monthly)
    event_times = rng.exponential(1 / hazards)

    # Draw dropout times (independent censoring)
    # Convert annual dropout rate to monthly hazard
    if assumptions.dropout_rate_annual > 0:
        dropout_hazard_monthly = -np.log(1 - assumptions.dropout_rate_annual) / 12
        dropout_times = rng.exponential(1 / dropout_hazard_monthly, n)
    else:
        dropout_times = np.full(n, np.inf)

    # Observed event time = min(event, dropout)
    # A patient who drops out before their event doesn't contribute an event
    observed_time = np.minimum(event_times, dropout_times)
    had_event = event_times <= dropout_times

    return observed_time, had_event, is_treatment


def run_single_simulation(assumptions: TrialAssumptions, rng: np.random.Generator) -> dict:
    """
    Run one simulated trial and return the calendar time (in months from
    trial start) when each analysis milestone is hit.
    """
    # Step 1: enrollment dates (months from trial start)
    enrollment_times = simulate_enrollment_dates(assumptions, rng)

    # Step 2: per-patient event times (months from their enrollment)
    time_to_event, had_event, _ = simulate_event_times(assumptions, rng)

    # Step 3: calendar time of each event = enrollment + time_to_event
    # Only patients who had an event (not dropouts) count
    event_calendar_times = enrollment_times + time_to_event
    event_calendar_times = event_calendar_times[had_event]
    event_calendar_times.sort()

    # Step 4: how many events needed for final analysis?
    total_events_needed = required_events(
        hr=assumptions.target_hazard_ratio,
        alpha=assumptions.alpha_two_sided,
        power=assumptions.power,
        ratio=assumptions.randomization_ratio,
    )

    # Step 5: find calendar time when each milestone is hit
    milestones = {}
    for frac in assumptions.information_fractions:
        events_needed = int(np.ceil(frac * total_events_needed))
        if events_needed <= len(event_calendar_times):
            milestones[frac] = event_calendar_times[events_needed - 1]
        else:
            # Not enough events ever accumulated in this sim run
            # (e.g., HR too close to 1, or accrual too slow)
            milestones[frac] = np.nan

    milestones["total_events_needed"] = total_events_needed
    milestones["enrollment_complete"] = enrollment_times.max()
    return milestones


def run_monte_carlo(assumptions: TrialAssumptions, n_simulations: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Run many simulated trials and return the distribution of milestone dates.

    Each row = one simulated trial. Columns = milestone times in months.
    The caller can then compute percentiles, plot histograms, etc.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_simulations):
        # Create a fresh child RNG per sim for reproducibility
        sim_rng = np.random.default_rng(rng.integers(0, 2**31))
        results.append(run_single_simulation(assumptions, sim_rng))

    df = pd.DataFrame(results)
    return df


def months_to_date(months: float, start: datetime) -> datetime:
    """Convert months-from-start to a calendar date."""
    if pd.isna(months):
        return None
    return start + timedelta(days=months * 30.44)  # average month length


def summarize_results(df: pd.DataFrame, assumptions: TrialAssumptions) -> pd.DataFrame:
    """
    Produce a summary table: for each milestone, show the P10/P50/P90
    predicted calendar dates.
    """
    summary_rows = []
    for frac in assumptions.information_fractions:
        if frac not in df.columns:
            continue
        values = df[frac].dropna()
        if len(values) == 0:
            continue
        summary_rows.append({
            "Milestone": f"{int(frac*100)}% Information Fraction",
            "Events required": int(np.ceil(frac * df["total_events_needed"].iloc[0])),
            "P10 date (optimistic)": months_to_date(values.quantile(0.10), assumptions.enrollment_start),
            "P50 date (median)": months_to_date(values.quantile(0.50), assumptions.enrollment_start),
            "P90 date (conservative)": months_to_date(values.quantile(0.90), assumptions.enrollment_start),
            "P50 months from start": round(values.quantile(0.50), 1),
        })
    return pd.DataFrame(summary_rows)
