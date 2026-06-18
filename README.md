# TimeStone AI

**Predict the outcome of a business transformation before you commit to it.**

> "See 1,000 futures. Choose the one truth."

TimeStone builds a synthetic **digital twin** of a company, generates up to **1,000 transformation scenarios** (Claude-powered, with a rule-based fallback), runs each through a **Monte Carlo simulation** with realistic risk, and ranks the strategies that maximize **risk-adjusted NPV** — so leaders decide with a probability distribution, not a single optimistic number.

---

## The problem

Transformation decisions — digital programs, M&A integration, market expansion, big tech bets — are made on one deterministic spreadsheet and a gut feel. Then most of them miss their targets. The spreadsheet never showed the downside, the execution risk, or the ramp-up curve.

TimeStone replaces the single number with a distribution of outcomes you can actually reason about.

---

## What you do with it

| Step | Example |
| --- | --- |
| 01 — Build the twin | Model the company's financials, operations, and market as a digital twin. |
| 02 — Generate scenarios | Produce up to 1,000 transformation hypotheses (Claude or rule-based). |
| 03 — Simulate | Run Monte Carlo on each scenario with shocks, delays, and adoption ramp. |
| 04 — Rank & stress-test | Sort by P(NPV>0), mean NPV, and payback; tornado-analyze the top picks. |

---

## TimeStone is for you if

- ✅ You're a transformation lead, strategy team, or operator weighing a major bet
- ✅ You're tired of single-point NPV that hides the downside
- ✅ You want execution risk, market shocks, and adoption ramp modeled explicitly
- ✅ You need a defensible, probability-weighted recommendation for the board

---

## Features

🏢 **Digital Twins** — Synthetic models of an organization across financials, operations, and market.

🎲 **Monte Carlo Engine** — 1,000 iterations per scenario with external shocks and ramp-up; deterministic seeding for reproducibility.

🧠 **Scenario Generation** — Claude generates transformation hypotheses; falls back to a rule-based generator with no API key.

📊 **Risk-Adjusted Ranking** — Strategies ranked by P(NPV>0), mean NPV, ROI multiple, and median payback.

🌪️ **Sensitivity / Tornado** — See which assumptions actually move the outcome.

🧪 **Invariant Test Suite** — pytest locks the financial-model invariants (NPV math, risk variance, no-perfect-success guardrail, seeding).

🖥️ **Interactive Dashboard** — Streamlit + Plotly UI plus a CLI for batch runs.

🏭 **Multi-Industry Templates** — Transportation, energy, fintech, SaaS, manufacturing out of the box.

---

## What's under the hood

```
1. DATA INPUT          -> Company financials, operations, market data
2. DIGITAL TWIN        -> Synthetic model of the business
3. SCENARIO GENERATION -> up to 1,000 hypotheses (Claude or rule-based)
4. MONTE CARLO         -> 1,000 iterations/scenario with shocks + ramp-up
5. SYNTHESIS API       -> Programmatic access to runs, scenarios, results
6. RANKING             -> Top-N by P(NPV>0), NPV, payback
7. SENSITIVITY         -> Tornado analysis on the top scenarios
```

**Financial model:** capex in year 0, benefits after a (delayed) implementation period, adoption ramp 40% / 70% / 95% / 100% by year, 5-year NPV at configurable WACC (default 12%).

**Risk modeled per iteration:** execution failure (~5%), market downturn (~8%, −30% revenue), competitive response (~15%, −20% revenue), cost overruns (up to +50%), implementation delays (up to +80%). All configurable via `SimulationConfig`.

---

## Example output (national rail operator — transportation)

Anonymized example. The model ships with several industry twins; company names are illustrative.

```
RANK #1  Dynamic Pricing Implementation
   P(NPV > 0)          : 96%
   Mean NPV (5y)       : $43M
   Mean ROI multiplier : 8.9x
   Median payback      : 2 years
   Recommendation      : PROCEED with phased rollout

RANK #3  Predictive Maintenance System
   P(NPV > 0)          : 64%
   Median payback      : 5 years
   Recommendation      : PILOT first — high execution risk
```

Example industry twins included: a **national rail operator** (transportation), a **consumer-fintech platform**, a **national power grid** (energy), a **B2B SaaS** company, and a **discrete manufacturer**.

---

## Tech stack

- **Language:** Python 3.10+
- **AI:** Anthropic Claude SDK (optional; rule-based fallback)
- **Simulation:** NumPy, Monte Carlo
- **Modeling:** dataclasses + Pydantic
- **API:** synthesis API + synthetic-data service for programmatic runs
- **UI:** Streamlit + Plotly
- **Testing:** pytest (financial-invariant suite)

---

## Quickstart

```bash
git clone https://github.com/westfellow25/timestone-ai.git
cd timestone-ai
pip install -r requirements.txt
cp .env.example .env          # optional: ANTHROPIC_API_KEY for AI scenarios

python -m timestone list-companies          # see available industry twins
python -m timestone assess "<company name>" # run a full assessment
streamlit run src/timestone/interfaces/web/dashboard.py
```

Run the tests:

```bash
pytest tests/ -v
```

---

## What TimeStone is not

- **Not a BI dashboard.** It simulates futures, it doesn't just chart the past.
- **Not a point forecast.** Every answer is a distribution with explicit risk.
- **Not a black box.** The financial model and risks are configurable and test-locked.

---

## Roadmap

- [x] Monte Carlo engine, digital twins, multi-industry templates
- [x] Claude scenario generation + rule-based fallback
- [x] Synthesis API + synthetic-data service
- [x] Invariant test suite
- [ ] PDF export of the executive report
- [ ] Real-time data integration (financial APIs)
- [ ] Multi-objective optimization (NPV vs risk vs time)
- [ ] Bayesian updating from pilot results

---

Built by [@westfellow25](https://github.com/westfellow25).
