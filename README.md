# Competitive Readout Simulator

Monte Carlo simulation of event-driven clinical trial timelines. Predicts when interim and final analyses will occur given enrollment, survival, and trial design assumptions.

## What it does

Given a set of trial assumptions (sample size, accrual rate, control median survival, target HR, etc.), runs thousands of simulated trials and returns the distribution of predicted readout dates for each information fraction milestone (e.g., IA1 at 50%, IA2 at 80%, final at 100%).

Instead of a single point estimate ("readout on March 15, 2028"), you get a probabilistic range ("80% chance of readout between Jan 2028 and July 2028").

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push this repo to GitHub (public or private)
2. Go to share.streamlit.io
3. Click "New app", select the repo, branch, and `app.py` as the entry point
4. Deploy

## Methodology

- **Required events** calculated via Schoenfeld's formula
- **Enrollment** modeled as linear ramp-up followed by steady-state uniform accrual
- **Survival times** drawn from exponential distribution (constant hazard)
- **Dropout** modeled as independent exponential censoring
- **Readout timing** = calendar time at which cumulative events reach the milestone threshold

## Roadmap

- [ ] Weibull survival for non-proportional hazards
- [ ] Scenario comparison (save and overlay multiple assumption sets)
- [ ] AI agent layer: auto-ingest ClinicalTrials.gov updates and ASCO abstracts, propose updated assumptions
- [ ] Real-world data integration for control arm benchmarks
