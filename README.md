# Tennis Predictor 🎾

A full-stack tennis prediction & betting intelligence platform covering ATP + WTA.

## Features

- **Pre-match predictions** — Win probability for any matchup
- **Set score predictions** — Full probability distribution (2-0, 2-1, etc.)
- **Live in-play updates** — Probabilities shift as sets complete
- **9 betting markets** — Match winner, set score, tiebreak, game/set handicap, and more
- **Lay/Back odds** — Model fair price vs market odds with value detection
- **Rich scorecard UI** — H2H, surface stats, form, Elo ratings

## Tech Stack

| Component | Service | Cost |
|-----------|---------|------|
| Frontend | Cloudflare Pages | $0 |
| Database | Cloudflare D1 | $0 |
| Storage | Cloudflare R2 | $0 |
| Cron Jobs | Cloudflare Workers | $0 |
| Prediction API | Railway (Hobby) | $5/mo |

## Project Structure

```
tennis-predictor/
├── data-pipeline/       # Phase 1: Data download, clean, Elo computation
├── models/              # Phase 2-3: ML training scripts & notebooks
├── backend/             # Phase 5: FastAPI prediction API (Railway)
├── frontend/            # Phase 6: React app (Cloudflare Pages)
└── .github/workflows/   # CI/CD
```

## Getting Started

### Phase 1: Data Pipeline

```bash
cd data-pipeline
pip install -r requirements.txt
python data_pipeline.py
```

Downloads 57 years of ATP + WTA match data, computes Elo ratings, and generates clean datasets.

### Phase 2: Train Model (coming soon)

### Phase 3: Simulation Engine (coming soon)

### Phase 4-7: Backend, Frontend, Deployment (coming soon)

## License

Private — All rights reserved.
