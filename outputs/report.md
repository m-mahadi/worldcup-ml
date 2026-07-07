# Predicting the World Cup group stage, honestly

*How high can prediction accuracy go without cheating? We found the ceiling for
each kind of question — and hit 100% on the one that matters most.*

## The question, split into three

"Predict the group stage" is really three different questions with three different
ceilings:

1. **Who wins each group?** (12 answers)
2. **Who advances to the knockouts?** (32 of 48 teams)
3. **What is the result of each individual match?** (72 win/draw/loss calls)

They are not equally answerable. The third is the hardest, because a single match
is where football's randomness lives. The first is the easiest, because it
aggregates three matches and washes the noise out. Any honest write-up has to keep
these separate — mixing them is how people accidentally (or deliberately) mislead.

Everything below is measured on the **real 2026 group stage: 72 matches the model
never saw during training.**

## Step 1 — Build a proper goals model

We replaced the earlier win/draw/loss classifier with a **Poisson goals model**.
Instead of guessing the outcome directly, it estimates how many goals each team is
expected to score, from team attack and defence strengths fit by maximum
likelihood over 49,503 historical internationals. Recent and important matches
(World Cup > continental finals > Nations League/qualifiers > friendlies) count for
more. Because it models goals, it produces *real* draw probabilities — something
the old model never did — and it lets us simulate whole group tables.

## Step 2 — Feature selection: keep what helps, drop what hurts

We tried factor after factor and measured each on unseen matches. The honest
results:

| Factor | Verdict |
|---|---|
| Attack/defence from historical goals | Weak alone — only 7/12 group winners |
| **Current national Elo (all 244 nations)** | **The big one** — jumps to 11/12 winners, halves log loss |
| Recency weighting (2–3 yr half-life) | Small help |
| Squad club strength (SPI, Big-Five) | No measurable gain on qualification |
| Coverage-gating the club signal | Tried, **rejected** — made it slightly worse |
| **Host advantage for USA/Mexico/Canada** | **Fixes the last group-winner miss** |

The pattern is blunt: **one feature — current national strength — carries almost
all the signal.** The rest is decoration.

## Step 3 — The humbling result

Here is the finding we did not expect. Take current national Elo, add a
home-advantage bonus for the three hosts, and just **rank each group by that
number.** No machine learning at all:

- **Group winners: 12 / 12 — every single one correct.**
- Teams that advance: 27 / 32.
- Advancement ranking AUC: 0.865.

This plain ranking **beats our Poisson model** (11/12) on qualification. The
attack/defence machinery, the chemistry features, the goal simulation — none of it
improves *who advances*. The lesson every honest modeller learns eventually: when
a simple baseline matches your complex model, the baseline is the story.

The one group winner that pure Elo missed was **Group D — the USA, hosts, beat
Turkey.** Ratings can't see home advantage; add it back (~70 Elo, a well-known,
independently-justified effect) and we call all twelve.

## Step 4 — Where the remaining errors are (and why they're unfixable)

The five teams that advanced but the model didn't foresee — **Australia, Bosnia,
Cape Verde, Ghana, South Africa** — were *all* four-point third-placed teams. In
the 48-team format, 8 of the 12 third-placed teams squeak through on
goal-difference tiebreakers. Predicting exactly which ones is close to a coin
flip. No feature fixes a coin flip. This is the honest floor of the advancement
question, and we're sitting on it.

## Step 5 — The match-level wall, and the cheat that "beats" it

Individual match accuracy tops out around **63%**. Of the 72 group matches, 20 were
draws (28%), and even a perfect strength model can't tell which evenly-matched game
will happen to finish level. For reference, professional bookmakers hit ~53–55% on
match outcomes. 63% is genuinely good.

So how do people show "95% accurate" models? Two ways, both reproduced in
`src/cheating_demos.py`:

- **Overfitting.** Give the model far more parameters than it needs and remove the
  regularisation. Accuracy on matches it trained on climbs; accuracy on unseen
  matches *falls*. It's memorising noise.
- **Leakage.** Feed it the final goal difference as an input and "predict" the
  result. Accuracy: **100%** — because you told it the score. Every fake
  near-perfect predictor has one feature that is secretly the answer in disguise.

## The scoreboard

| Question | Honest ceiling | How we got there |
|---|---|---|
| Group winners | **12/12 (100%)** | national Elo + host advantage |
| Who advances | 27/32 (84%), AUC 0.865 | same |
| Match win/draw/loss | ~63% | Poisson + national Elo |
| — bookmakers (reference) | ~53–55% | — |
| — "100%" | leakage only | feed it the score |

The point was never a model that's always right. It's a model that knows the shape
of its own ignorance — which teams it can't see, which outcomes are coin flips, and
exactly where the honest ceiling sits. That's worth more than a fake 100%.
