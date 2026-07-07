# worldcup-ml

An honest 2026 World Cup prediction project.

The goal is not to claim a magic bracket. It is to find which questions a model can
answer well, and where football becomes too noisy for a single-match prediction to
be trusted.

## Results

| Target | Result |
|---|---:|
| Group winners | 12 / 12 |
| Teams that advanced from groups | 27 / 32 |
| Match win/draw/loss accuracy | about 63% |
| Blind Round of 32 picks | 14 / 16 |
| Knockout walk-forward picks | 19 / 22 |

The main finding is simple: current national strength plus travel-adjusted home
advantage is enough to call the group winners. Richer quality signals, especially
xG, matter more once the knockouts start.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python src/run.py
PYTHONPATH=src python src/run_rolling.py
PYTHONPATH=src python src/run_knockout.py
PYTHONPATH=src python src/make_bracket_svg.py
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:PYTHONPATH = "src"
python src/run.py
python src/run_rolling.py
python src/run_knockout.py
python src/make_bracket_svg.py
```

## Repository map

```text
data/raw/
  international_results.csv       historical national-team results
  eloratings_world.tsv            current national Elo snapshot
  worldcup2026_api_games.json     World Cup fixtures and results
  worldcup2026_api_stadiums.json  host stadium metadata
  match_xg_stats.csv              match-level xG data
  squad_players.csv               squad/player data
  spi_club_rankings.csv           club-strength ratings

src/
  model.py             data loading, Elo, Poisson goals model
  geo.py               distance-based home advantage
  evaluate.py          group-stage and validation scoring
  knockout.py          knockout rating updates and simulations
  quality.py           xG and squad-quality adjustments
  run.py               group-stage model comparison
  run_knockout.py      blind knockout bracket prediction
  run_rolling.py       walk-forward knockout evaluation
  make_bracket_svg.py  bracket visualization
  cheating_demos.py    overfitting/leakage demos

outputs/
  metrics.csv
  report.md
  knockout_report.md
  knockout_bracket.csv
  champion_probabilities*.csv
  predicted_bracket.svg
```

## Data sources

- Historical international results from public match-result datasets.
- National Elo ratings from eloratings.net.
- 2026 World Cup fixtures, venues, and played results.
- Match xG from Sofascore.
- Club-strength ratings from SPI club rankings.
- Squad/player lists from official World Cup squad data.

## Notes on interpretation

- Single-match result prediction tops out around the low 60s because draws and
  knockout variance are genuinely hard to predict.
- Group winners are easier because three matches smooth out some luck.
- Penalty shootouts are treated as high-variance outcomes, not as something the
  model should pretend to know.
- The leakage demo in `src/cheating_demos.py` exists to show why perfect-looking
  sports models are usually using information from after the match.

## Write-up

- Public essay: `BLOG.md`
- Technical report: `outputs/report.md`
- Knockout report: `outputs/knockout_report.md`
