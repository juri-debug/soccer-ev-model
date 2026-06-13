# Football Predictor

A statistical + machine-learning football match predictor with full tournament simulation. Covers the top-5 European leagues, every international team, and the 2026 FIFA World Cup with the actual final draw, real fixture dates, and squad-strength priors.

## What it does

- **Single match**: predict any matchup with outcome probabilities, expected goals, top scorelines, and a full scoreline heatmap.
- **League season**: simulate the rest of a Premier League / La Liga / Bundesliga / Serie A / Ligue 1 season from current standings - probability table for title win, top 4, top 6, relegation.
- **Tournament**: World Cup 2026, World Cup 2022, Euro 2024, AFCON, Copa America - real draws when applicable. Includes group-stage Predicted fixtures organised by date, knockout bracket cascade, predicted champion, and full Monte Carlo aggregate simulations.

## Model stack

Per-match probabilities come from an ensemble:

1. **Elo** - goal-difference weighted, competition-aware K-factors
2. **Pi-rating** (Constantinou-Fenton 2013) - separate home/away strength per team
3. **Dixon-Coles** - time-decayed bivariate Poisson with low-score correlation
4. **XGBoost + LightGBM + CatBoost** - 28 engineered features (multi-window form, streaks, head-to-head, rest, rolling xG)
5. **Isotonic-calibrated logistic regression meta-learner** - blends all four base models, optionally including bookmaker odds and WC 2026 squad-strength priors

Test metrics (held-out chronological split, n=4,334 league matches, n=3,780 internationals).

**Aggregate** — accuracy and Ranked Probability Score (RPS, lower is better):

| Model | Acc | RPS | Bootstrap 95% CI on RPS |
|---|---|---|---|
| Dixon-Coles alone | 0.490 | 0.212 | [0.207, 0.216] |
| CatBoost alone | 0.521 | 0.201 | [0.197, 0.205] |
| Stacked ensemble (no odds) | 0.526 | 0.203 | [0.200, 0.206] |
| **Stacked + bookmaker odds** | **0.531** | **0.198** | **[0.194, 0.202]** |
| B365 bookmaker (baseline) | 0.536 | 0.195 | — |

**Honest caveat on the headline**: the bookmaker's RPS (0.195) sits inside the stacked-model's 95% CI [0.194, 0.202]. The two are statistically indistinguishable on this split. Without odds as a feature, CatBoost alone is within noise of the full stacked ensemble, so the stacking layer's marginal contribution is small.

**Per-league RPS (where the variation lives)**:

| Competition | n | RPS |
|---|---|---|
| La Liga | 941 | 0.193 |
| Serie A | 938 | 0.194 |
| Bundesliga | 765 | 0.197 |
| Premier League | 935 | 0.199 |
| Ligue 1 | 755 | 0.208 |

~8% spread between best and worst league. Ligue 1 is genuinely harder to predict than La Liga.

**Internationals by competition**:

| Competition | n | RPS |
|---|---|---|
| FIFA WC qualification | 901 | 0.150 |
| UEFA Euro qualification | 239 | 0.151 |
| AFCON | 104 | 0.177 |
| Friendlies | 967 | 0.183 |
| UEFA Nations League | 243 | 0.191 |
| **Aggregate internationals** | 3,780 | **0.173** |

Internationals look better than leagues mostly because qualification matches contain a lot of mismatches (mid-tier vs minnow) that any model gets right.

**What this doesn't yet test**: walk-forward retraining across distinct regimes (pre-COVID / COVID / VAR-era / post-pandemic) to check whether the small edge holds when the model has to predict each new era from data preceding it. That's the next robustness check on the roadmap.

## Data sources

All free, no paid APIs:
- **football-data.co.uk** - 28k+ league matches with 11 bookmakers' odds
- **github.com/martj42/international_results** - 25k+ international results since 2000
- **Understat** (via `soccerdata`) - rolling xG features for top-5 leagues
- **ESPN squad rankings + bookmaker outright odds** - WC 2026 squad-strength priors

## Run locally

```bash
pip install -r requirements.txt
python train.py            # fetches data + trains both scopes (~1 min)
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Fork this repo (or push your fork to your GitHub)
2. Go to https://share.streamlit.io
3. Sign in with GitHub, click "New app"
4. Pick the repo, set main file to `app.py`
5. Deploy. You'll get a public URL.

Daily retraining is handled by the GitHub Action at `.github/workflows/daily-update.yml` - it fetches fresh data and commits updated models, which auto-redeploys the app.

## Project layout

- `src/` - data fetchers, ratings (Elo, Pi), Dixon-Coles fit, features, predictor, tournament sim
- `train.py` - end-to-end training pipeline with stacked ensemble + calibration
- `app.py` - Streamlit UI (Single match / League season / Tournament tabs)
- `data/processed/` - cached parquet files used at runtime
- `models/` - trained predictor bundles (joblib)
