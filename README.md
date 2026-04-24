**Author: [Julie Hohenberg]([(https://www.linkedin.com/in/juliehohenberg/)])**
# Competitive Readout Simulator

A Monte Carlo simulation tool that predicts when a clinical trial will announce its results ("read out"). Built for competitive intelligence and strategy teams who need to anticipate competitor timelines.

Instead of giving a single predicted date, the tool runs thousands of simulated trials and returns a probability distribution — e.g., "80% chance of final readout between January and July 2028."

---

## Table of contents

1. [Quick start](#quick-start)
2. [What problem this solves](#what-problem-this-solves)
3. [How to use the app](#how-to-use-the-app)
4. [The math, explained](#the-math-explained)
5. [Assumptions and limitations](#assumptions-and-limitations)
6. [Roadmap](#roadmap)

---

## Quick start

**Local development:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Streamlit Cloud:**
1. Push this repo to GitHub
2. Go to share.streamlit.io, click "New app"
3. Select the repo and set `app.py` as the entry point
4. Deploy

---

## What problem this solves

Clinical trials in oncology (and many other disease areas) don't have a fixed end date. They end when enough **events** have happened — an "event" being something like disease progression (for PFS endpoints) or death (for OS endpoints). The more events, the more statistical evidence the trial has accumulated.

This means the readout date depends on things we can only estimate:

- How fast the trial enrolls patients
- How sick the patients are (faster disease = events happen sooner)
- How much the drug actually helps (bigger effect = events happen later in the treatment arm)
- How many patients drop out

Because all of these inputs are uncertain, the output — the readout date — is also uncertain. A point estimate like "readout in Q2 2028" hides that uncertainty. A distribution reveals it.

---

## How to use the app

The sidebar has three groups of inputs:

**Enrollment** — When does the trial start? How many patients total? How fast do they enroll? Is there a randomization ratio (1:1, 2:1, etc.)?

**Survival assumptions** — How long does the control arm typically survive (median months)? What's the assumed treatment effect (hazard ratio)? What's the dropout rate?

**Analysis plan** — What's the significance threshold (alpha) and target power? At what information fractions do the interim analyses occur (e.g., 50%, 80%, 100%)?

The main panel shows:
- **Required events** for the final analysis (calculated from your inputs)
- **Summary table** with P10, P50, P90 predicted dates for each milestone
- **Histogram** showing the full distribution of predicted readout dates
- **Sensitivity analysis** showing how readout timing shifts as the hazard ratio changes

---

## The math, explained

This section walks through every equation used, defines each variable, and gives intuition for what they represent.

### 1. How many events does the trial need?

The trial keeps running until it accumulates enough events to reliably detect the assumed treatment effect. That "enough" is defined by **Schoenfeld's formula**:

$$d = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{p_1 \cdot p_2 \cdot (\ln(HR))^2}$$

**Variables:**

| Symbol | Meaning | Typical value |
|---|---|---|
| $d$ | Number of events required for the final analysis | e.g., 247 |
| $\alpha$ | Significance threshold (chance of a false positive) | 0.05 |
| $1 - \beta$ | Power (chance of detecting a real effect) | 0.80 or 0.90 |
| $z_{1-\alpha/2}$ | Z-score for the alpha level (from the normal distribution) | 1.96 for α=0.05 |
| $z_{1-\beta}$ | Z-score for the power level | 0.84 for 80% power |
| $HR$ | Target hazard ratio (assumed treatment effect) | 0.70 |
| $p_1$ | Proportion of patients in the treatment arm | 0.5 for 1:1 |
| $p_2$ | Proportion of patients in the control arm | 0.5 for 1:1 |

**Intuition:** The harder the effect is to detect, the more events you need. Three things make detection harder:
- A smaller true effect (HR closer to 1)
- A stricter significance threshold (smaller α)
- A higher power target (smaller β)
- An unbalanced randomization (uneven $p_1$ and $p_2$)

The $\ln(HR)^2$ in the denominator is the key driver. An HR of 0.70 gives $\ln(0.70)^2 = 0.127$. An HR of 0.85 gives $\ln(0.85)^2 = 0.026$ — about **5× smaller**, which means you need about 5× as many events to detect a smaller effect.

**Worked example:** HR = 0.70, α = 0.05, power = 80%, 1:1 randomization:

$$d = \frac{(1.96 + 0.84)^2}{0.5 \cdot 0.5 \cdot (\ln 0.70)^2} = \frac{7.84}{0.25 \cdot 0.127} \approx 247 \text{ events}$$

### 2. When does each patient get enrolled?

Real trials don't enroll at full speed on day 1. Sites activate gradually. We model this as a **linear ramp-up** followed by steady-state accrual.

**During ramp-up** (months 0 to $T_r$), the enrollment rate increases linearly from 0 to the steady-state rate $r$. The number of patients enrolled during ramp-up is the area under that triangle:

$$N_{rampup} = \frac{1}{2} \cdot r \cdot T_r$$

**Variables:**

| Symbol | Meaning |
|---|---|
| $r$ | Steady-state monthly accrual rate (patients/month) |
| $T_r$ | Ramp-up duration (months) |
| $N_{rampup}$ | Patients enrolled during the ramp-up phase |

To generate a random enrollment time during the ramp-up phase, we use **inverse CDF sampling** from a triangular distribution. If $u$ is a uniform random number between 0 and 1:

$$t_{enroll} = T_r \cdot \sqrt{u}$$

The square root skews the samples later in the ramp — which is correct, because enrollment is faster at the end of ramp-up than at the start.

**After ramp-up**, patients enroll at rate $r$ until the sample size $N$ is reached. Enrollment times in this phase are drawn from a uniform distribution between $T_r$ and $T_r + (N - N_{rampup})/r$.

### 3. When does each patient have their event?

For each patient, we draw a **time-to-event** from an exponential distribution. The exponential distribution assumes a **constant hazard** — meaning the instantaneous risk of an event is the same every month.

The exponential distribution has one parameter: the hazard rate $\lambda$. For a patient in the control arm:

$$\lambda_{control} = \frac{\ln 2}{m_{control}}$$

**Variables:**

| Symbol | Meaning |
|---|---|
| $\lambda_{control}$ | Monthly hazard rate for control arm (instantaneous event rate) |
| $m_{control}$ | Median survival in control arm (months, user input) |
| $\ln 2$ | ≈ 0.693, comes from the median of an exponential distribution |

**Why $\ln 2 / m$?** The median of an exponential distribution with rate $\lambda$ is $\ln(2) / \lambda$. So if we know the median, we can solve for $\lambda$.

**Worked example:** If the control median is 12 months, then $\lambda_{control} = 0.693 / 12 \approx 0.058$. That means roughly a 5.8% chance per month of having an event.

For the treatment arm, the hazard is multiplied by the hazard ratio:

$$\lambda_{treatment} = \lambda_{control} \cdot HR$$

An HR of 0.70 means the treatment arm has 70% of the control arm's monthly event risk. Events happen more slowly in the treatment arm, which is why readout takes longer when the drug actually works.

**Drawing a random event time:** Given a hazard $\lambda$ and a uniform random number $u$:

$$t_{event} = -\frac{\ln(1-u)}{\lambda}$$

This is inverse CDF sampling again. NumPy's `rng.exponential(1/λ)` does this under the hood.

### 4. When does a patient drop out?

Dropout is modeled as **independent censoring** with its own exponential distribution. If the annual dropout rate is $d_{ann}$, the monthly hazard is:

$$\lambda_{dropout} = -\frac{\ln(1 - d_{ann})}{12}$$

**Variables:**

| Symbol | Meaning |
|---|---|
| $d_{ann}$ | Annual probability of dropping out (user input, e.g., 0.05 = 5%/year) |
| $\lambda_{dropout}$ | Monthly hazard of dropout |

**Why this formula?** If the annual dropout probability is 5%, then the probability of *not* dropping out in a year is 95%. Under a constant-hazard model, the probability of surviving 12 months is $e^{-12\lambda_{dropout}}$. Setting $e^{-12\lambda_{dropout}} = 0.95$ and solving gives the formula above.

**What counts as an event?** For each patient, we draw both a time-to-event and a time-to-dropout. The patient contributes an event to the trial **only if the event happens before the dropout**. If the dropout comes first, the patient is "censored" and doesn't count toward the event tally.

### 5. Finding the readout date

For one simulated trial:

1. Generate enrollment times $e_i$ for all $N$ patients
2. For each patient, generate a time-to-event $t_i$ (or censor them if dropout comes first)
3. Compute the **calendar time** of each event: $c_i = e_i + t_i$
4. Sort all event calendar times in ascending order
5. The readout at information fraction $f$ happens when the $(f \cdot d)$-th event occurs

**Variables:**

| Symbol | Meaning |
|---|---|
| $e_i$ | Enrollment time of patient $i$ (months from trial start) |
| $t_i$ | Time from enrollment to event for patient $i$ (months) |
| $c_i$ | Calendar time of patient $i$'s event (months from trial start) |
| $f$ | Information fraction (e.g., 0.5 = interim analysis at 50% of events) |
| $d$ | Total events needed for final analysis (from Schoenfeld) |

So if $d = 247$ and $f = 0.5$, the interim analysis happens at the moment the 124th event occurs.

### 6. Monte Carlo: repeat and summarize

Steps 1–5 describe **one** simulated trial. But each simulation has randomness baked in (random enrollment times, random event times, random dropouts). Run it once and you get one answer. Run it 1,000 times and you get a distribution.

After running $n$ simulations (default 1,000), we compute summary statistics from the distribution of readout calendar times:

- **P10** — the 10th percentile: only 10% of simulations finished earlier
- **P50** — the median: 50% of simulations finished earlier, 50% later
- **P90** — the 90th percentile: only 10% of simulations took longer

This gives the CSI team a defensible probabilistic range instead of a single date that implies false precision.

---

## Assumptions and limitations

**Proportional hazards** — This simulator assumes the treatment arm's hazard is the control hazard multiplied by a constant HR. This is true for cytotoxic chemotherapy but often wrong for immunotherapy (checkpoint inhibitors, CAR-T), where treatment effects can be delayed or have crossing survival curves. If you're modeling a non-proportional-hazards scenario, the predictions will be biased.

**Exponential survival** — We assume constant hazard over time. Real survival curves often have time-varying hazards (e.g., high early mortality that decreases over time). A Weibull distribution would be more flexible; this is on the roadmap.

**Independent censoring** — We assume dropout is independent of prognosis. In reality, sicker patients may be more likely to drop out, which violates this assumption.

**No protocol amendments** — We assume enrollment and analysis plans don't change mid-trial. Real trials amend protocols frequently.

**Single-site pooling** — Accrual is modeled as one global rate. Multi-regional trials with different activation timelines aren't separately modeled.

**Use for directional estimates, not regulatory submissions.** This tool is designed for competitive intelligence — rough timeline forecasting with appropriate uncertainty. It should not be used as a substitute for formal trial design software (EAST, PASS, rpact).

