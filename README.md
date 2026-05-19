# TimeStone AI

**Predict business transformation outcomes before you commit.**

TimeStone AI uses synthetic digital twins and Monte Carlo simulations to forecast the success probability, NPV and payback of business transformations under realistic uncertainty.

> "See 1,000 futures. Choose the one truth."

---

## What is TimeStone AI?

TimeStone creates a **digital twin** of a company, generates **up to 1,000 transformation scenarios** (Claude-powered when an API key is set, rule-based otherwise), runs **Monte Carlo simulations** with realistic risk modeling, and ranks the strategies that maximize risk-adjusted NPV.

**Use cases:**

- Digital transformation strategy
- M&A integration planning
- Product launch decisions
- Market expansion analysis
- Technology adoption roadmaps

---

## How it works

```
1. DATA INPUT          -> Company financials, operations, market data
2. DIGITAL TWIN        -> Synthetic model of the business
3. SCENARIO GENERATION -> 1,000 transformation hypotheses (Claude or rule-based)
4. MONTE CARLO         -> 1,000 iterations per scenario with shocks & ramp-up
5. RANKING             -> TOP-N by P(NPV>0), NPV, payback period
6. SENSITIVITY         -> Tornado analysis on top scenarios
```

---

## Example output (KTZ - National Railway)

```
RANK #1  Dynamic Pricing Implementation
   P(NPV > 0)          : 96%
   Mean NPV (5y)       : $43M
   Mean ROI multiplier : 8.9x
   Median payback      : 2 years
   90% CI ROI          : [4.1x, 14.6x]
   Recommendation      : PROCEED with phased rollout

RANK #2  AI-Powered Route Optimization
   P(NPV > 0)          : 94%
   Mean NPV (5y)       : $32M
   Median payback      : 2 years
   Recommendation      : PROCEED, monitor adoption

RANK #3  Predictive Maintenance System
   P(NPV > 0)          : 64%
   Mean NPV (5y)       : $4M
   Median payback      : 5 years
   Recommendation      : PILOT first - high execution risk
```

---

## Tech stack

- **Language:** Python 3.10+
- **AI:** Anthropic Claude (`anthropic` SDK) - optional, falls back to rule-based
- **Simulation:** NumPy, Monte Carlo with NPV
- **Visualization:** Streamlit, Plotly
- **Data modeling:** dataclasses + Pydantic
- **Testing:** pytest

---

## Installation

```bash
git clone https://github.com/westfellow25/timestone-ai.git
cd timestone-ai

pip install -r requirements.txt

# (Optional) configure Claude API for AI-generated scenarios
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick start

```bash
# 1. Build a digital twin of KTZ (or any company)
python -m src.models.digital_twin

# 2. Generate transformation scenarios (rule-based, no API key needed)
python -m src.simulation.scenario_generator

# 2b. Or use Claude for creative scenarios (requires ANTHROPIC_API_KEY)
python -m src.simulation.claude_scenarios

# 3. Run Monte Carlo simulations
python -m src.simulation.monte_carlo

# 4. Sensitivity analysis on a specific scenario
python -m src.simulation.sensitivity

# 5. Launch the interactive dashboard
streamlit run src/api/dashboard.py
```

---

## Tests

```bash
pytest tests/ -v
```

The test suite locks in the core financial-model invariants (NPV math, risk-level variance, no-perfect-success guardrail, deterministic seeding).

---

## Financial model

- Capex paid in year 0
- Benefits start after the (delayed) implementation period
- Adoption ramps up: **40% / 70% / 95% / 100%** by year
- `cost_reduction` applies to **operating costs** (not revenue)
- 5-year NPV at configurable WACC (default 12%)

External risks modeled per iteration:

- **Execution failure:** 5% prob (project lost)
- **Market downturn:** 8% prob (-30% to revenue impact)
- **Competitive response:** 15% prob (-20% to revenue impact)
- **Cost overruns:** up to +50% for high-risk projects
- **Implementation delays:** up to +80% for high-risk projects

All defaults are configurable through `SimulationConfig`.

---

## Roadmap

- [x] Core simulation engine with NPV, ramp-up, external shocks
- [x] Digital twin modeling
- [x] Multi-industry templates (transportation, energy, fintech, SaaS, manufacturing)
- [x] Interactive dashboard with sensitivity analysis
- [x] Claude-powered scenario generation (optional)
- [x] Test suite locking in financial invariants
- [ ] PDF export of executive report
- [ ] Real-time data integration (financial APIs)
- [ ] Multi-objective optimization (NPV vs risk vs time)
- [ ] Bayesian updating from pilot results

---

## License

MIT - see `LICENSE`.

---

Built by [@westfellow25](https://github.com/westfellow25) | 2026
