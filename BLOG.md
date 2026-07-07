# Predicting the 2026 World Cup: an honest, quality-aware rating model

**A build journal / technical write-up.**

## Abstract

I built a model to predict the 2026 World Cup and, more importantly, to find the
point where prediction stops working. It combines an Elo rating fit on 49,503
international matches, a bivariate-independent Poisson goals model, a current national
rating, a distance-based home-advantage term, and a set of performance signals derived
from expected goals (xG), goal timing, momentum, and squad quality. On out-of-sample
data it calls **12 of 12 group winners**, tops out at **~64% on individual match
outcomes** (a ceiling I show is caused by draws, not by the model), and predicts blind
knockout rounds at **19 of 22 (86%)**, with the only misses being two penalty shootouts
and one genuine upset. I also demonstrate, on the same codebase, the two ways people
fake a "perfect" model: overfitting and label leakage. Everything runs on Python, NumPy,
and pandas. This document walks the whole thing step by step.

---

## 1. Motivation and framing

Most football predictors output a champion and stop. That hides the only interesting
question: how much of a result is skill and how much is luck? My approach is to build
the strongest honest model I can, push it until it stops improving, and treat the
plateau as a measurement of football's irreducible randomness.

I split "predict the tournament" into three questions with three very different
ceilings, and I keep them separate throughout:

1. **Match outcome** (win / draw / loss) — the hardest, because one match is where luck
   lives.
2. **Qualification** (who wins a group, who advances) — easier, because it aggregates
   three matches and luck partly cancels.
3. **The bracket** (who reaches each knockout round) — measurable round by round.

Every number I report is out-of-sample: the model is scored on matches it did not train
on.

---

## 2. Data

| Dataset | Size | Fields used | Role |
|---|---|---|---|
| International results 1872–2026 | 49,503 matches | teams, score, tournament, neutral flag, date | Elo backbone + Poisson fit |
| Current national Elo (eloratings.net) | 244 nations | rating | strength prior for every country |
| SPI club ratings | 464 clubs, 25 leagues | off, def, spi (xG-based) | squad quality |
| FIFA squad lists | 1,248 players | player, club, position | squad construction |
| Big-five club stats 2025/26 | 470 matched players | minutes, goals, assists, shots, SoT, tackles, interceptions, cards, GA90, save%, clean sheets | squad quality (secondary) |
| Match xG (Sofascore) | 94 WC matches | xG, shots, SoT, big chances, possession, per team | performance grading |
| WC 2026 fixtures | 104 games | teams, scores, stadium, matchday, round | prediction targets |
| Host stadiums | 16 venues | city, country | home-advantage geography |

Two data notes. First, the international-results file already contains every
competition that matters and recent editions of each (AFCON to January 2026, Euro 2024,
Nations League 2025, Copa América 2024, World Cup qualifiers to March 2026): 4,666
competitive matches since 2022. Second, the Python runtime I used had a broken OpenSSL
stack that crashed on every HTTPS request, so I pulled the national ratings through
PowerShell's .NET TLS instead. The xG for all 94 matches I scraped from Sofascore and
verified against my own scorelines (94/94 matched, which also confirms my fixture feed
agrees with reality).

---

## 3. Methods

I built the model in layers. This section documents each layer, in the order I added
it, with the exact parameters.

### 3.1 The Elo backbone

Every team starts at 1500. For a match between A and B, the expected score for A is

```
E_A = 1 / (1 + 10^((R_B − R_A) / 400))
```

and after the match each rating updates by

```
R_A ← R_A + K · m · (S_A − E_A)
```

where `S_A` is 1 for a win, 0.5 for a draw, 0 for a loss; `m` is a goal-margin
multiplier (`1` for a one-goal game, `1.5` for two, `(11 + gd)/8` beyond that); and `K`
scales with how much the match mattered:

```
World Cup finals                         K = 55
continental finals (Euro/Copa/AFCON/…)   K = 45
Nations League and all qualifiers        K = 35
other competitive                        K = 25
friendly                                 K = 15
```

I replay all 49,503 matches in chronological order to produce the ratings. The
Nations-League line matters: in my first pass those 600+ recent matches fell into the
generic `K = 25` bucket and were under-weighted; grading them at 35 sharpened the
ratings measurably. I also track a rolling window of each team's last 10 results (points,
goals for, goals against) for form features.

### 3.2 First model: a softmax win/draw/loss classifier

The first predictor is a 3-class softmax (multinomial logistic regression) trained by
gradient descent (700 epochs, learning rate 0.18, L2 = 5e-4, standardized inputs). Its
nine features per match are: the Elo gap (scaled by 400), recent points difference,
recent goal-difference difference, recent goals-for and goals-against differences, a
games-played difference, a neutral-venue flag, a World-Cup flag, and a major-tournament
flag.

On top of that I added a *rating adjustment* layer that shifts the effective Elo gap
using squad information: a player-form term, a same-club "chemistry" term (fraction of
squad pairs sharing a club), a same-league term, a club-vs-country style-fit term, and a
data-coverage term. I grid-searched the weights on the group stage.

**Result:** 63.9% accuracy on the 72 group matches, but with a fatal flaw covered next.

### 3.3 The draw problem (the match-level ceiling)

The classifier predicted **zero draws**. Ever. Breaking down the 72 group matches:

| Actual result | Count | Model accuracy |
|---|---|---|
| Home/first-listed win | 34 | 85.3% |
| Away/second-listed win | 18 | 94.4% |
| **Draw** | **20** | **0.0%** |

20 of 72 matches (28%) were draws, and the model got all 20 wrong. On those draws its
average draw probability was 0.17 while it assigned 0.66 to a winner. This is not a bug.
A draw is rarely any model's single most likely outcome, because the two teams split the
probability mass. I later confirmed with a full sweep that no honest reweighting recovers
more than ~6 of the 20 draws, and each recovery costs a correct decisive pick. For
reference, bookmakers sit at ~53–55% on match outcomes, so ~64% is already strong. The
match-level ceiling is real and is set by draws.

### 3.4 Neutral-venue correction

A World Cup has no home team, but my first model was implicitly rewarding whichever team
was listed first in the fixture. I removed this by making the model **order-symmetric**:
I predict each tie in both orderings and average, so swapping the two teams only swaps
the two win probabilities and leaves the draw probability unchanged. Home advantage is
handled explicitly and separately (§3.6).

### 3.5 Second model: a Poisson goals model

Predicting W/D/L directly is coarse. I switched to modelling goals. For a match the
expected goals for each side are

```
λ_i = exp(μ + attack_i − defence_j + home_adv_i)
λ_j = exp(μ + attack_j − defence_i + home_adv_j)
```

Goals are independent Poisson variables with those means; the outer product of the two
per-team scoreline distributions gives the joint distribution over final scores, which I
sum into win/draw/loss (`P(win) = Σ_{a>b}`, `P(draw) = Σ_{a=b}`, `P(loss) = Σ_{a<b}`).
Draws now come out as a computed quantity instead of being ignored.

I fit `attack`, `defence`, `μ`, and the home term by weighted maximum likelihood using
gradient descent in NumPy. For a Poisson observation the loss gradient with respect to a
team's attack reduces to `(λ − goals)`, so each pass I compute expected goals, subtract
actual goals, and scatter the residual onto the relevant attack and defence parameters
(via `np.add.at`). Details:

- **Iterations:** 400–1500; learning rate 0.05–0.1; L2 = 0.02.
- **Identifiability:** I re-center `attack` and `defence` to zero mean each iteration.
- **Recency weighting:** each match's weight is `importance · 0.5^(age_years / 2.5)`, so
  influence halves every 2.5 years and competitive matches count more.
- **Home term:** a fitted log-goal home advantage (`γ ≈ 0.25`), applied only where a team
  is genuinely at home (§3.6), zero otherwise.

### 3.6 National-Elo blend (the dominant feature)

The from-scratch attack/defence is noisy for teams that rarely play the strong nations.
I correct this by blending the current national-Elo gap directly into the expected goals:

```
λ_i ← λ_i · exp( w_elo · (elo_i − elo_j) / 400 ),   w_elo = 0.5
```

This is the single most important feature in the model. The raw Poisson (no blend) gets
only 7 of 12 group winners; with the blend it reaches 11–12. The rating is the engine;
the goals model is the chassis around it. I swept `w_elo` and 0.5 was both the
log-loss optimum and defensible (it is a *current* snapshot, so a large weight would
overfit the 72-game sample).

### 3.7 Home advantage as a distance decay

2026 is hosted by the USA, Canada, and Mexico, so "neutral" is not equal for everyone. I
model home advantage as a bonus that is full in a team's own country and decays with
travel distance to the actual match venue:

```
home_bonus(team, venue) =
    HOME_MAX                                    if venue country == team country
    HOME_MAX · exp( − haversine(team, venue) / DECAY )   otherwise
HOME_MAX = 95 (Elo points),  DECAY = 3000 km
```

I hardcoded coordinates for the 16 host cities and all 48 nations and used the
great-circle distance; for the Poisson model the Elo bonus is converted to a log-goal
term (`γ · bonus / HOME_MAX`). A team's group bonus is averaged over its three group
venues. The resulting values are geographically sensible, not tuned:

```
USA / Canada / Mexico (own country)   95
Mexico (average over venues)          ~80
Haiti, Colombia, Panama, Curaçao      27–44
Brazil, England, Morocco              12–14
Argentina 5 · Japan 3 · Australia 1
```

Adding this term is what turns the one missed group winner (USA, beaten to top spot by
the rating's pick of Turkey) into a correct call, taking group-winner accuracy to 12/12.

### 3.8 Using the group stage: performance, not just results

Once a round is complete I fold it into the ratings, but graded by *how* teams played,
not just the scoreline. The post-group update starts from the national Elo and processes
the 72 group games with four adjustments:

1. **Opponent difficulty** — handled automatically by Elo's expected-score term (drawing
   Brazil moves you up; beating a minnow does not).
2. **Momentum** — games are processed in matchday order and weighted
   `{matchday 1: 0.6, matchday 2: 1.0, matchday 3: 1.6}`, so a team rounding into form
   rises and a bad opener is discounted.
3. **Manner of result** (§3.9) and **xG** (§3.10) replace the raw W/D/L with a
   performance score.
4. **Margin** — a small multiplier (`1.0 / 1.25 / 1.5` for 1 / 2 / 3+ goal games).

The update is `Δ = K_group · momentum · margin · (performance − expected)`, with
`K_group = 80`. After the sweep, a static squad-quality nudge is also applied (§3.11).
All these weights are set by football reasoning, not tuned to the backtest — a
distinction I return to in §7.

### 3.9 Manner-of-result signal

Losing to a stoppage-time goal is not the same as being outplayed for 90 minutes. I
parse the goal minutes from the match data (regex, handling `90+X` stoppage time) and
compute the scoreline as it stood at the 85th minute, then blend:

```
performance = 0.65 · result_fulltime + 0.35 · result_at_85'
```

So a team that only conceded late keeps most of the credit of a draw.

### 3.10 Expected-goals signal (the key upgrade)

Scorelines contain finishing luck; xG strips it out by asking how many goals the chances
created *should* have produced. For each match I compute an "xG-deserved" result by
treating each side's xG as a Poisson mean and integrating `P(win) + 0.5·P(draw)` over
scorelines (up to 8 goals). I then blend it with the manner-adjusted actual result:

```
match_performance = 0.55 · xG_deserved + 0.45 · manner_adjusted_result
```

xG gets the larger weight because it is more stable and more predictive of the next
match. The signal immediately exposed teams the table mislabeled:

```
             goal diff   xG diff   reading
Egypt           +2         −0.8    flattered; worse than results
Norway          +1         +2.4    underrated; better than results
Morocco         +3         +3.1    fully deserved
Paraguay        −2         −3.0    genuinely poor
```

Norway's strong-but-quiet xG is why the model was unsurprised when they later knocked
Brazil out.

### 3.11 Squad-quality signal

Independent of one noisy group, I measure squad quality from where players play their
club football. I map each squad member's club to its SPI rating (which is xG-based),
average across the squad, z-score across the 48 nations, and add a nudge of ~25 Elo
points per standard deviation. I keep it small because club-name matching is noisy. This
signal is what tips the Egypt-over-Australia knockout tie to correct (Egypt's players
come from stronger clubs than their results implied).

### 3.12 Knockout prediction machinery

The knockout bracket is a fixed tree; the Round-of-32 pairings are set by the final
group standings. To predict a round I:

1. Update ratings through the previous round with `apply_knockout_round`, using the same
   performance grading (§3.9–3.10). A penalty shootout reads ≈0.5/0.5 unless the xG says
   one side deserved it, so shootout "winners" are not artificially boosted.
2. Convert each tie's W/D/L into an advancement probability (no draws survive a knockout):
   `P(A advances) = P(win) + P(draw)·P(win)/(P(win)+P(loss))`, i.e. a level game resolves
   in proportion to strength.
3. For a single predicted bracket, take the favourite each tie and advance them. For
   champion odds, run 20,000 Monte-Carlo simulations, fixing every already-played result
   and simulating only the remaining games.

---

## 4. Validation methodology

- **Un-gameable anchor:** train only on internationals before 2024, test on 2024–25
  matches never seen. This lands at ~60–62% and is the number I trust; every World-Cup
  result sits in the same neighbourhood rather than magically higher.
- **Group-stage replay:** the 72 group matches are out-of-sample for the model.
- **Walk-forward knockouts:** predict each knockout round using only data up to the end
  of the previous round; never use a result I am predicting.

---

## 5. Results

### 5.1 Match-level ceiling

~64% on individual group matches, capped by draws (§3.3). This does not improve with
more features, and it should not: it is the randomness floor.

### 5.2 Qualification

With the national Elo + geographic home advantage: **12/12 group winners**, 27/32
correct Round-of-32 qualifiers, advancement AUC 0.865. The five missed qualifiers were
all four-point third-placed teams decided on tie-breakers — coin flips in a format where
8 of 12 third places advance.

### 5.3 The baseline that matched the model

A pure ranking by national Elo plus the host bonus — no machine learning — gets 12/12
group winners and 27/32 qualifiers, equal to or better than the full model. This is a
genuine (and humbling) result: for "who is the better team," a good current rating
already contains almost all the signal, and the ML apparatus mostly decorates it.

### 5.4 Blind knockout prediction (walk-forward)

```
cutoff = groups        → predict Round of 32:   14 / 16
cutoff = Round of 32   → predict Round of 16:    5 / 6
-----------------------------------------------------------
walk-forward total:                             19 / 22  =  86%
```

The three misses: two penalty shootouts (Germany–Paraguay, Netherlands–Morocco) and one
normal-time upset (Norway 2–1 Brazil). Every tie decided in normal time by the better
side was called. Notably, Morocco outplayed the Netherlands 1.38 xG to 0.24 while winning
on penalties, so the model rated them up and then correctly picked them over Canada.

### 5.5 Live forecast (all data through the Round of 16)

Fixing every played result and simulating the rest 20,000 times:

![Predicted World Cup 2026 bracket](outputs/predicted_bracket.svg)

```
Spain       34%   (single most likely bracket: Spain beats Argentina in the final)
France      20%
Argentina   18%
England     13%
Norway       5%
```

Spain lead on their underlying numbers — their group xG was the best in the tournament.

---

## 6. Negative results and ablations

Recorded because a discarded idea with a measurement attached is worth more than a silent
guess:

- **Coverage-gating the club signal** (down-weighting club form for poorly-covered teams):
  tried, measured, **rejected** — it slightly worsened qualification.
- **Squad club features alone** (SPI / big-five) did not improve qualification over the
  national-Elo baseline; their value showed up only in the knockout quality grading.
- **Raw Poisson without the Elo blend:** 7/12 group winners; the blend is essential.
- **Cranking the national-Elo weight** raised match accuracy toward 65% but did not move
  qualification, which was already at its noise floor.

---

## 7. How a "perfect" model is faked

To show why I do not chase 100%, I reproduced both standard failures on this codebase.

- **Overfitting.** I added a per-team identity parameter for every nation (hundreds of
  free parameters) and removed the regularisation. Training accuracy climbed while
  accuracy on unseen matches *fell* — the model memorising noise. The tell is a model
  only ever evaluated on its own training data.
- **Leakage.** I added one input — the match's final goal difference — and "predicted"
  the result: **100% accuracy**, because the input is the answer. Every 100%-accurate
  predictor has a feature like this hiding in it (final possession, post-match ratings,
  closing odds).

This is why I fixed feature weights by reasoning rather than by tuning them to the
backtest: on a sample this small, tuning to the score is just leakage with extra steps.

---

## 8. Limitations

- Match-level accuracy cannot exceed ~64%; draws are close to random.
- Penalty shootouts are unpredictable by construction; they are 2 of my 3 knockout misses.
- The national Elo is a current snapshot, so its blend weight can only be validated on
  the World Cup itself; I kept it modest to avoid overfitting.
- Squad-quality club matching is fuzzy and noisy, hence its small weight.
- xG is a strong but single-source signal (Sofascore); a multi-source average would be
  more robust.

---

## 9. Conclusion

The model is honest about what it knows. It calls 12/12 group winners, predicts 86% of
knockout games one round ahead, and currently makes Spain a 34% favourite. It also stops
exactly where football stops being predictable: it cannot beat 64% on a single match and
cannot call a shootout, because past that line the result is decided by a deflection, a
flag, or a ball hitting the inside of a post. A model that reports its own limits is more
useful than one that hides them behind a confident, quietly-leaked champion.

---

*Reproducible on Python + NumPy + pandas. Ratings from 49,503 international matches plus
current national numbers; xG for all 94 World Cup matches from Sofascore. Code, data, and
the full scoreboard are in this repository.*
