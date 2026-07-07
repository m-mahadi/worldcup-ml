# Predicting the knockouts, blind

*Data cutoff: the end of the group stage. Everything below is predicted without
looking at a single knockout result.*

## The setup

We froze the model's knowledge at the moment the group stage ended. Two pieces of
new information became available at that instant, and we used both — honestly:

1. **Who actually advanced, and how they played.** We took the 72 real group
   results and updated every team's rating with them (an Elo update weighted for
   World Cup importance and margin of victory). Teams that over-performed —
   Morocco, Norway — climbed; teams that stumbled slipped.
2. **The bracket.** The Round-of-32 match-ups are fixed by the final group
   standings, and the slot-tree (which winner meets which) is structural. Both are
   group-stage facts, not results.

Then we predicted R32 → Round-of-16 → Quarter-finals → Semi-finals → Final,
advancing our *own* predicted winners at every step. No knockout score was ever
read by the model.

## How the blind Round of 32 did

**13 of 16 correct — 81.2%.**

| Result | Match | We picked | Actually advanced |
|---|---|---|---|
| ✅ | South Africa v Canada | Canada | Canada |
| ❌ | Germany v Paraguay | Germany | Paraguay *(pens)* |
| ❌ | Netherlands v Morocco | Netherlands | Morocco *(pens)* |
| ✅ | Brazil v Japan | Brazil | Brazil |
| ✅ | France v Sweden | France | France |
| ✅ | **Ivory Coast v Norway** | **Norway** | **Norway** *(upset called)* |
| ✅ | Mexico v Ecuador | Mexico | Mexico |
| ✅ | England v DR Congo | England | England |
| ✅ | United States v Bosnia | United States | United States |
| ✅ | Belgium v Senegal | Belgium | Belgium |
| ✅ | Portugal v Croatia | Portugal | Portugal |
| ✅ | Spain v Austria | Spain | Spain |
| ✅ | Switzerland v Algeria | Switzerland | Switzerland |
| ✅ | Argentina v Cape Verde | Argentina | Argentina |
| ✅ | Colombia v Ghana | Colombia | Colombia |
| ❌ | Australia v Egypt | Australia | Egypt |

All three misses were upsets, and **two were decided by penalty shootouts** — a
coin toss the model has no way to call. The one upset the model *did* foresee
(Norway over Ivory Coast) it got right. 81% on a single-elimination round is strong;
for reference, the favourite wins a knockout tie only ~65–70% of the time.

## The predicted bracket

- **Quarter-finalists:** France, Spain, England, Argentina
- **Final:** Spain vs Argentina
- **Champion (most-likely bracket): Spain** — third place France

## Champion odds (20,000 blind simulations)

| Team | Champion | Reaches final | Reaches semi |
|---|---:|---:|---:|
| Argentina | 21.2% | 37.5% | 56.8% |
| Spain | 20.2% | 31.3% | 49.3% |
| France | 19.9% | 32.6% | 53.5% |
| England | 9.5% | 18.9% | 35.5% |
| Colombia | 4.7% | 12.0% | 23.7% |
| Portugal | 3.3% | 7.0% | 15.7% |
| Netherlands | 3.1% | 7.4% | 17.0% |
| Mexico | 2.7% | 7.5% | 18.2% |

## Reading it honestly

The title is a **genuine three-way toss-up** — Argentina, Spain, and France sit
within ~1.5% of each other. Note the split between the two views: the *most-likely
single bracket* crowns Spain (they have the highest chance of winning each specific
tie on their path), but *across all simulations* Argentina edges ahead because
their side of the draw is slightly softer. Both are correct answers to different
questions: "who wins the likeliest path" vs "who is most likely to win the whole
thing."

This is the nature of knockout football: single elimination is maximum variance.
Our group-stage prediction could hit 100% on group winners because three matches
smooth out luck. A knockout tie is one match, sometimes one penalty shootout — so
the honest output isn't a confident champion, it's a tight cloud of three.

*Regenerate: `PYTHONPATH=src python src/run_knockout.py`*
