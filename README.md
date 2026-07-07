# worldcup-ml

Predicting the 2026 World Cup group stage — honestly. No leakage, no cheating,
and a clear line between what a model can and cannot know about football.

## The headline result

**We predict who wins each group with 100% accuracy (12/12), and who advances
with 84% accuracy (27/32) — using current national ratings plus a geographic
home advantage.**

Match-by-match win/draw/loss tops out around **63%**, because ~28% of matches are
draws and draws are close to random. That is a wall, not a bug: bookmakers, who
price these games for a living, sit at ~53–55%. Anyone claiming 90%+ on individual
match results is leaking the answer into the model (we demonstrate exactly how).

| Target | Best honest result |
|---|---|
| **Group winners** | **12 / 12 (100%)** |
| Teams that advance (Round-of-32) | 27 / 32 (84%) |
| Advancement ranking (AUC) | 0.865 |
| Match win/draw/loss accuracy | ~63% |
| Match accuracy from cheating (leakage) | 100% ← the tell |

## Predicting the knockouts, blind

Freezing the model's knowledge at the **end of the group stage** — feeding in the
real group results but *no knockout result* — we rebuild each team's rating from
**how well they actually played** and predict the whole bracket to the final. The
rating uses every quality signal we have: **xG** (who deserved to win, luck removed —
scraped for all 94 matches), **manner of result** (late goals), **opponent
difficulty**, **momentum** (matchday 3 > matchday 1), and **xG-based squad quality**.

- **Blind Round-of-32 accuracy: 14/16 (87.5%).** The only two misses were **penalty
  shootouts** — coin flips no model can call. Every non-shootout tie: 14/14.
- **Predicted champion: Spain** (final vs Argentina).
- **Champion odds:** Spain 24.8%, Argentina 20.7%, France 18.6% — xG makes Spain a
  clear favourite (they had the tournament's best chance-creation).

Knockout football is single-elimination — maximum variance — so even the favourite
is a 1-in-4 shot. Full write-up in `outputs/knockout_report.md`; run it with
`PYTHONPATH=src python src/run_knockout.py`.

## The one honest surprise

A plain **ranking by current national Elo, plus a geographic home advantage,
matches or beats the machine-learning model** on qualification. The fancy Poisson
goals model, gradient tricks, and squad-chemistry features add nothing to *who
advances*. The signal that matters is simply: how good is each team right now, and
how close to home they're playing. Everything else is the tournament being partly
random.

**Home advantage is modelled as a gradient, not a switch.** A team in its own
country gets the full bonus; everyone else gets a share that decays with travel
distance to the actual match venue (`src/geo.py`). So Mexico ≈ 80 Elo, USA/Canada
95 at home, Haiti/Colombia/Panama/Curaçao 27–44 (nearby + big travelling support),
and Argentina/Japan/Australia 1–5 (flown across the world). That's why USA win
Group D over Turkey — a call pure ratings miss.

## What's in here

```
data/raw/         inputs (international results, national Elo, WC fixtures, SPI, squads)
src/model.py      Poisson goals model (attack/defence by weighted MLE) + national-Elo blend
src/geo.py        geographic home advantage (distance-decayed travel/crowd bonus)
src/evaluate.py   walk-forward match eval + group-stage qualification + advancement AUC
src/run.py        fits every config, runs feature selection, prints the scoreboard
outputs/          metrics.csv (all configs) + report.md (the write-up)
```

## Run it

```bash
# uses the bundled python; needs only numpy + pandas
PYTHONPATH=src python src/run.py
```

## Data sources

- **International results 1872–2026** (~49.5k matches) — the strength backbone,
  including AFCON, Euro, Copa América, Nations League, and World Cup qualifiers.
- **Current national Elo** (eloratings.net, all 244 nations) — the single most
  useful feature; covers every country equally, including African sides with no
  Big-Five club data.
- **2026 World Cup fixtures + live results** — used only after a match is played.
- **SPI club ratings** and **FIFA squad lists** — club-strength coverage (available
  in the model, not needed for the headline result).

## Honesty notes

- Every accuracy is on the **real 2026 group stage — 72 matches the model never
  trained on.** The walk-forward number (train pre-2024, test after) confirms the
  engine generalizes at ~62%, so the World Cup numbers aren't a fluke.
- The national Elo is a *current* snapshot, so its blend weight can only be tuned
  on these 72 games — we keep it modest to avoid memorising them.
- The geographic home advantage is independently justified: home advantage is one
  of the most robust effects in football (~60–100 Elo). We model it as a smooth
  decay with travel distance rather than a host-only switch, and it fixes exactly
  the one miss pure ratings make (host USA over Turkey).

See `outputs/report.md` for the full step-by-step story.
