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
# Install (editable mode)
pip install -e .

# List available digital twins
python -m timestone list-companies

# Run a full assessment (scenarios + Monte Carlo + report)
python -m timestone assess "Kazakhstan Temir Zholy (KTZ)"

# Browse past runs
python -m timestone list-runs

# Launch the interactive dashboard
streamlit run src/timestone/interfaces/web/dashboard.py
```

See `docs/architecture.md` for the project's layered architecture.

---

## Deploy

**Streamlit Community Cloud (free, 30 seconds):**

1. Push to a public GitHub repo
2. Go to https://share.streamlit.io
3. New app -> point at `streamlit_app.py` in this repo
4. Click Deploy. URL like `https://timestone.streamlit.app`

**Static landing page:**

`site/index.html` is a single-file landing page (no build step). Drop it on any
static host (GitHub Pages, Vercel, Netlify, Cloudflare Pages) and point a
domain like `timestone.ai` at it.

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
