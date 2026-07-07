# Predicting the knockouts, blind — with a quality-of-team model

*Data cutoff: the end of the group stage. Everything below is predicted without
looking at a single knockout result.*

## The setup

We froze the model's knowledge at the moment the group stage ended, then rebuilt
each team's rating from **how well they actually played** — not just their results —
and predicted R32 → Final, advancing our own picks at every step. No knockout score
was ever read by the model.

The rating carried into the knockouts folds in every quality signal we could get:

| Signal | What it captures | Source |
|---|---|---|
| **xG (expected goals)** | who *deserved* to win on the balance of chances, luck stripped out | Sofascore, all 94 matches |
| **Manner of result** | losing to a 90+' goal ≈ a draw's worth of performance | goal minutes |
| **Opponent difficulty** | drawing Brazil ≫ beating a minnow | Elo expected score |
| **Momentum** | rounding into form: matchday 3 counts more than matchday 1 | matchday weights |
| **Squad quality** | a squad of xG-strong clubs is strong regardless of one noisy group | SPI club ratings (xG-based) |

Each group result updates a team's rating by *performance* — an xG-deserved result
blended with the manner-adjusted score — so a team that was outplayed but won on a
lucky goal rises less than one that dominated the shot charts.

## How the blind Round of 32 did

**14 of 16 correct — 87.5%.**

| Result | Match | We picked | Actually advanced |
|---|---|---|---|
| ✅ | South Africa v Canada | Canada | Canada |
| ❌ | Germany v Paraguay | Germany | Paraguay *(penalties)* |
| ❌ | Netherlands v Morocco | Netherlands | Morocco *(penalties)* |
| ✅ | Brazil v Japan | Brazil | Brazil |
| ✅ | France v Sweden | France | France |
| ✅ | **Ivory Coast v Norway** | **Norway** | **Norway** *(upset called — xG loved Norway)* |
| ✅ | Mexico v Ecuador | Mexico | Mexico |
| ✅ | England v DR Congo | England | England |
| ✅ | United States v Bosnia | United States | United States |
| ✅ | Belgium v Senegal | Belgium | Belgium |
| ✅ | Portugal v Croatia | Portugal | Portugal |
| ✅ | Spain v Austria | Spain | Spain |
| ✅ | Switzerland v Algeria | Switzerland | Switzerland |
| ✅ | Argentina v Cape Verde | Argentina | Argentina |
| ✅ | Colombia v Ghana | Colombia | Colombia |
| ✅ | **Australia v Egypt** | **Egypt** | **Egypt** *(quality model flipped this)* |

**The only two misses were penalty shootouts.** Germany and Netherlands were the
better sides — on results *and* on xG — and lost the lottery. No model can, or
should, call a shootout. Everything a model *can* know, this one got right: 14 of
14 non-shootout ties.

The quality signals earned their keep: adding xG, manner, momentum and squad
quality turned the Australia–Egypt call correct (Egypt's players come from stronger
clubs than their Elo showed) and confidently backed the Norway upset (their group
xG was far better than their results).

## The predicted bracket

- **Quarter-finalists:** France, Spain, England, Argentina
- **Final:** Spain vs Argentina
- **Champion: Spain** — third place France

## Champion odds (20,000 blind simulations)

| Team | Champion | Reaches final | Reaches semi |
|---|---:|---:|---:|
| **Spain** | **24.8%** | 36.3% | 53.9% |
| Argentina | 20.7% | 38.7% | 58.7% |
| France | 18.6% | 30.5% | 53.6% |
| England | 8.6% | 18.5% | 34.9% |
| Colombia | 4.0% | 10.9% | 22.4% |
| Portugal | 3.6% | 7.4% | 15.9% |
| Brazil | 3.5% | 9.1% | 19.8% |
| Netherlands | 2.9% | 6.9% | 17.2% |

## Update — moving the cutoff forward (walk-forward)

As the tournament advances we move the data cutoff forward one round at a time and
predict the *next* round blind, folding each completed round's results (with xG)
into the ratings. This is the honest test: predict, then reveal.

| Cutoff | Predicts | Blind accuracy |
|---|---|---|
| End of group stage | Round of 32 | **14 / 16** |
| End of Round of 32 | Round of 16 | **5 / 6** |
| **Combined walk-forward** | every knockout game so far | **19 / 22 = 86%** |

The three total misses: **two penalty shootouts** (R32) and **one on-day upset**
(Norway 2–1 Brazil in the R16). Everything decided in normal time by the better
side, the model called. Note the R32 xG did its job: Morocco *outplayed* Netherlands
on xG (1.38 vs 0.24) despite winning on penalties, so the model rightly rated them
up and then correctly picked **Morocco over Canada** in the R16.

**Latest prediction, from the end-of-R32 ratings:**

- Remaining Round-of-16 (still to play): Argentina and Colombia to advance
- Quarter-finalists: France, Spain, England, Argentina
- Final: **Spain vs Argentina → champion Spain**

| Team | Champion | Reaches final | Reaches semi |
|---|---:|---:|---:|
| **Spain** | **26.4%** | 37.9% | 58.7% |
| France | 22.6% | 36.8% | 67.4% |
| Argentina | 18.5% | 37.4% | 59.2% |
| England | 7.5% | 17.2% | 32.7% |
| Brazil | 5.0% | 13.3% | 28.9% |
| Morocco | 3.9% | 9.3% | 25.7% |

*(`PYTHONPATH=src python src/run_rolling.py`)*

## Reading it honestly

Feeding in *quality* rather than just results sharpened the picture. Before xG, the
title was a flat three-way tie (~20% each); now **Spain leads at 24.8%**, because
their group xG (+4.7 goal difference of chances) was the best in the tournament —
they were even more dominant than their scoreline said. Argentina still reaches the
final most often (softer side of the draw), but Spain is the most likely to lift it.

Two honest limits remain, and they are not data problems:

1. **Shootouts are coin flips.** Both R32 misses were penalties. That's the floor.
2. **Single elimination is high-variance.** Groups could hit 100% because three
   matches smooth out luck; one knockout tie can turn on a deflection. So even the
   favourite is only a 1-in-4 shot, and that's the honest number — not false
   confidence.

*Regenerate: `PYTHONPATH=src python src/run_knockout.py`. xG data: `data/raw/match_xg_stats.csv` (Sofascore, 94/94 matches, scraped via Codex).*
