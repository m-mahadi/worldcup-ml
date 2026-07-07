# I tried to predict the World Cup honestly, and the interesting part was where it broke

I wanted to know one thing: how much of a football match is skill and how much is
luck. Everyone builds a "World Cup predictor," slaps a champion on it, and moves on.
That's boring and it's usually a lie. The honest version is harder and way more
interesting: build the best predictor you can, push it until it stops getting better,
and the exact spot where it stalls tells you where the game stops being about who's
better and starts being about a coin flip.

So that's what I did. I built the thing from scratch, fed it decades of real matches,
scraped expected-goals for every game of this World Cup, and made it predict rounds
it had never seen. This is the whole thing written out. The data, the models, the
math, the weights, what worked, what embarrassed me, and where the wall is.

Fair warning, it's long and it gets technical. That's the point.

## The whole model in one line

Every prediction is the same question asked over and over: *who is better right now,
and by how much?* Everything below is me arguing with myself about how to measure
"better," and being honest about when knowing who's better still isn't enough to know
who wins.

## The data I actually used

You can't predict anything without a memory, so first I built one.

**Every international match from 1872 to today. 49,503 games.** Home team, away team,
score, competition, whether it was on neutral ground. This is the backbone. It's
where raw team strength comes from. And it already contains the stuff that matters:
the last African Cup of Nations, Euro 2024, Copa América 2024, the Nations League,
and the World Cup qualifiers. 4,666 competitive games since 2022 alone.

**Current national ratings for all 244 countries**, pulled from eloratings.net. This
matters more than it sounds, and I'll explain why later, but the short version is
that my from-scratch rating is noisy for teams that don't play much, and this fills
that hole. (Side note: the Python I was running had a broken SSL stack that crashed
on every https call, so I pulled this through PowerShell instead. Real-world data
work is 30% janitorial.)

**SPI club ratings for 464 clubs across 25 leagues.** These are xG-based
attack/defence numbers per club. I used them to measure how good a national squad
actually is, based on where its players play their club football. Covers the leagues
the big-five data misses: MLS, Saudi, Turkey, the lot.

**The official FIFA squad lists, 1,248 players**, plus big-five league club stats for
the 470 of them who play in England, Spain, Germany, Italy, or France.

**Expected goals for all 94 World Cup matches played so far**, scraped off Sofascore.
This is the big one for the knockout stage and I'll get to why.

**The fixtures, the results, and the coordinates of all 16 host stadiums**, so I
could do geography properly.

## Model 1: Elo, and why the competition matters

The strength backbone is an Elo rating. Same idea as chess. Everyone starts at 1500.
You beat someone, you take points off them; how many depends on how surprising the
result was. Beat a team way above you, big jump. Beat a minnow, barely moves.

The expected result of a game is

```
expected_A = 1 / (1 + 10^((rating_B - rating_A) / 400))
```

and after the match you nudge the rating by `K * (actual - expected)`, where `actual`
is 1 for a win, 0.5 for a draw, 0 for a loss. I replayed all 49,503 games in order to
get everyone's number.

Here's the part most people get lazy about: **`K` should depend on how much the game
mattered.** A friendly is not a World Cup knockout. Teams experiment in friendlies,
rest players, try nonsense. So I graded the importance:

```
World Cup finals      K = 55
continental finals    K = 45   (Euro, Copa, AFCON, Asian Cup, Gold Cup)
Nations League / quals K = 35
other                 K = 25
friendly              K = 15
```

I also scaled `K` up for blowouts (a 4-0 says more than a 1-0). The Nations League bit
actually mattered. In my first pass it was getting dumped into the "other" bucket,
which quietly threw away 600+ recent, high-quality games. Fixing that alone made the
ratings sharper.

## The first wall, and it's the whole story

I built a straightforward classifier on top of Elo. Nine features (rating gap, recent
form, goals for and against, that kind of thing), a softmax that spits out three
numbers: chance of a home win, a draw, an away win. I trained it, then replayed all 72
group matches it had never seen and made it call every one.

**64%.** Then I threw everything at it. More features, more data, cleverer math.

It stayed at 64%.

Here's why, and this is the soul of the whole project. Of those 72 group games, **20
were draws.** More than a quarter. And on those exact 20 games, the model's average
guess for "draw" was **17%**, while it was confidently backing a winner at **66%**.

The model wasn't broken. Draws are just close to random. When two even teams play, a
draw is the single most likely outcome, but the model almost never *picks* it, because
the two teams keep splitting the odds between them and the draw pile never wins the
three-way vote. You cannot know in advance which specific even game finishes level. It
depends on a deflection, a linesman's flag, a keeper having a day.

Put a number on how good 64% is. Professional bookmakers, with billions riding on
being right, land around 53-55% on match outcomes. So 64% is genuinely good. Which
means anyone showing you a model that's "92% accurate at picking match winners" is
either fooling themselves or feeding it the answer. I'll prove that at the end by
doing exactly that on purpose.

So I stopped trying to win every match and asked a question that can actually be
answered.

## The question that can be answered: who advances

One match is a weighted coin. But who *wins a group of four* is three matches, and
luck starts cancelling out. A strong team can draw a game, even lose one, and still
top the group. This is answerable.

And then football humbled me. I'd built this whole machine-learning apparatus. Out of
curiosity I tried the dumbest possible baseline: rank each group by one number, the
current national rating, call the top team the winner. No learning, no features, one
column sorted.

**It got 11 of 12 group winners.** My fancy model didn't beat it.

That's the lesson you only learn by measuring. When your one-line baseline matches
your clever model, the baseline *is* the finding. For "who is the better team," a good
rating already knows nearly everything. The machine learning was decorating a signal
that was already there.

The one group the rating missed was Group D. It said Turkey. The USA won it. And the
reason it missed is the next feature.

## Home advantage is a dimmer, not a switch

First, kill a bug. A World Cup has no home team; every game is at a neutral stadium.
My first model was secretly handing a bonus to whichever team got listed first in the
fixture. That's not home advantage, that's alphabetical luck. I ripped it out and made
the model symmetric: swap the two teams and the only thing that changes is which win
probability is which. The draw probability doesn't move.

But 2026 is in the USA, Canada, and Mexico, so *some* teams are more at home than
others, and this is real. Mexico in Guadalajara has 80,000 screaming fans, no jet lag,
familiar heat and altitude. Australia in New York flew 15,000 km into a strange time
zone to play in front of nobody. Treating those two as equally "neutral" is stupid.

So I modelled it as a dimmer. A team playing in its own country gets the full bonus.
Everyone else gets a slice of it that decays with how far they travelled to the actual
stadium:

```
home_bonus(team, venue) = 95 * exp(- distance_km / 3000)      (full 95 if it's their own country)
```

I hardcoded coordinates for the 16 host cities and all 48 countries and used the
great-circle distance. Here's what fell out, in rating points:

```
USA / Canada / Mexico (at home)   95
Mexico (across all its venues)    ~80
Haiti, Colombia, Panama, Curaçao  27-44   (close, huge travelling support, similar climate)
Brazil, England, Morocco          12-14
Argentina                          5
Japan                              3
Australia                          1
```

That's not me tuning knobs to win. That's geography. Nearby CONCACAF and South
American sides get a real edge; teams flown in from the other side of the planet get
basically nothing. Add that dimmer back in and the USA edge past Turkey.

**Now the rating calls all 12 group winners. 100%.** The only thing standing between a
good rating and a perfect dozen was remembering who's playing at home.

## Model 2: a proper goals model

Picking win/draw/loss directly is crude. What I actually want is to model *how many
goals each team is expected to score*, because that gives me real draw probabilities
and lets me simulate scorelines. So I switched to a Poisson model.

The idea: every team has an attack strength and a defence strength. The expected goals
for team *i* against team *j* is

```
λ_i = exp( μ + attack_i - defence_j + home_adv )
```

and the actual goals are drawn from a Poisson distribution with that mean. Multiply
the two teams' scoreline distributions together and you get the probability of every
final score, which you sum into win/draw/loss. Draws come out naturally, because
`P(1-1) + P(0-0) + P(2-2) + ...` is a real number the model computes instead of
ignoring.

I fit the attack and defence numbers by maximum likelihood, with plain gradient
descent in NumPy. The gradient is clean: for a Poisson, the derivative of the loss
with respect to a team's attack works out to `(λ - goals)`. So each pass I compute
every team's expected goals, subtract the actual goals, and shove the attack and
defence numbers in the direction that closes the gap. A few hundred iterations and it
settles.

Two things I weighted deliberately:

- **Recency.** A game from 2019 shouldn't count as much as one from this year. Each
  match gets a weight of `0.5 ^ (age_in_years / 2.5)`, so the influence halves every
  two and a half years.
- **The national-Elo blend.** The from-scratch attack/defence is noisy, especially for
  teams that don't play the big nations often. So I nudge each team's expected goals by
  the current-Elo gap: `λ *= exp( 0.5 * (elo_i - elo_j) / 400 )`. That 0.5 is the blend
  weight, and it's doing a lot of work. On its own the raw goals model got 7 of 12
  group winners. With the Elo blend it jumped to 11-12. The rating is the engine; the
  goals model is the chassis.

## Using the group stage: not just who won, but how

Once the group stage finished, I had new information, and the naive move is to just
look at the table. Points and goal difference. But that throws away most of what
happened. I graded each group result four ways.

**Opponent difficulty.** This one's free, because Elo already handles it. Drawing
Brazil moves your rating up; beating a minnow barely moves it. The expected-score
formula bakes strength of schedule in automatically. Morocco drew Brazil in their
opener, and the rating *knew* that was a strong result.

**Momentum.** A team that loses its first game then wins its next two is rounding into
form. That's more dangerous going into the knockouts than a team that started hot and
faded. So I processed the group games in matchday order and weighted them
`{matchday 1: 0.6, matchday 2: 1.0, matchday 3: 1.6}`. A bad opener costs you less; a
strong finish counts more. Paraguay got hammered 1-4 in their opener, but that was
matchday 1, so it got discounted.

**The manner of the result.** Losing to a 93rd-minute winner means you were level for
92 minutes. You played fine. So I parsed the goal minutes out of the match data and
computed the scoreline as it stood at the 85th minute, then blended that with the
full-time result (65% full-time, 35% at-85). A late collapse and a comfortable win no
longer count the same.

**Squad quality.** Independent of one noisy group, how good are these players? I took
each squad, looked up every player's club in the SPI ratings (which are xG-based), and
averaged. A team built from strong clubs is strong even if it stumbled for three games.
I added this as a small rating nudge, about 25 points per standard deviation of squad
quality, kept small on purpose because the club-name matching is noisy.

I set every one of these weights by reasoning about the football, not by tuning them to
make the backtest look good. That distinction is the whole difference between modelling
and lying to yourself, and I'll hit it again at the end.

## The best signal I added: expected goals

A scoreline lies. A team can dominate, hit the post four times, and lose to one
deflection. Another can get battered and nick a 1-0. If you rate teams on results, you
rate them partly on luck. xG is the fix. It asks: given the quality of the chances a
team actually created, how many goals *should* they have scored? It's the closest thing
football has to measuring who deserved to win.

I scraped xG for all 94 matches and folded it into the rating update. Instead of using
the raw win/draw/loss, I compute an "xG-deserved" result: treat each team's xG as a
Poisson mean and work out the probability they'd have won that game on the balance of
chances (`P(win) + 0.5 * P(draw)`). Then I blend it with the manner-adjusted actual
result, 55% xG, 45% actual. The xG gets the bigger vote because it's more stable and
more predictive of the next game.

And it immediately told me things the table hid:

```
                goal diff   xG diff
Egypt              +2         -0.8     -> flattered. worse than they looked.
Norway             +1         +2.4     -> underrated. better than they looked.
Morocco            +3         +3.1     -> fully deserved. genuinely strong.
Paraguay           -2         -3.0     -> genuinely poor.
```

Egypt got results but their underlying numbers were negative; they'd been a bit lucky.
Norway's results were ordinary but their chance creation was excellent. And here's the
payoff: when Norway later knocked Brazil out of the knockouts, the xG wasn't surprised.
It had been quietly telling me Norway were underrated the whole time.

## How I checked it honestly

The number I trust most isn't from the World Cup at all. I trained the model only on
matches before 2024 and tested it on 2024-25 games it had never seen. It landed around
60-62%. That's the honest "how good is the engine" number, the one that can't be gamed,
and every World Cup result I report sits in the same neighbourhood instead of magically
higher. If a step had suddenly claimed 85%, I'd have gone hunting for the leak.

## Then I predicted the knockouts blind

This is where it got fun, because now I could test against a real future. I froze the
model's knowledge at the end of the group stage and made it predict the entire knockout
bracket, Round of 32 to the final, without looking at a single knockout result.

**Round of 32: 14 of 16 correct.**

Look at the two misses. Germany lost to Paraguay. The Netherlands lost to Morocco. Both
went to **penalty shootouts.** A shootout is a coin flip that happens to involve feet,
and no model on Earth should claim to call one. Of the 14 ties decided in normal time
by the better team, I got all 14. That's not luck. That's the ceiling, and the ceiling
is honest.

The xG earned its keep here too. Egypt-over-Australia flipped to correct once squad
quality and xG were in, because Egypt's players come from better clubs than their raw
rating knew. And here's my favourite detail: Morocco *outplayed* the Netherlands 1.38
xG to 0.24 while winning on penalties. The model saw the performance, not the shootout,
rated Morocco up, and then correctly picked them to beat Canada in the next round.

## Rolling it forward

As the tournament moved, I moved the cutoff with it. Use everything through the Round of
32, predict the Round of 16 blind. Then use everything through the Round of 16, predict
the rest.

```
groups        -> Round of 32:   14 / 16
Round of 32   -> Round of 16:    5 / 6
------------------------------------------
walk-forward total:             19 / 22  =  86%
```

Three misses across the entire knockout stage. Two penalty shootouts and one honest
on-the-day upset, Norway beating Brazil in normal time. Everything the better side won
on the pitch, I called.

## Where it stands right now

Using everything that's actually happened, every group game, the Round of 32, and the
finished Round of 16, I fix those as facts and simulate the rest of the bracket 20,000
times. Here's the full bracket with the predicted path through the games still to play:

![Predicted World Cup 2026 bracket](outputs/predicted_bracket.svg)

```
Spain       34%   champion
France      20%
Argentina   18%
England     13%
Norway       5%
```

Spain sit clear at the top, and not because of their results, because of their *xG*,
which was the best in the tournament. The single most likely bracket has Spain beating
Argentina in the final.

## The two ways to fake a perfect model

I promised I'd show you the cheating, so I ran both tricks on my own model.

**Overfitting.** Give a model far more knobs than it needs and let it memorise the
training data. I did this deliberately: piled on parameters and stripped out the
regularisation. Its accuracy on games it had already seen climbed toward the sky, while
its accuracy on *new* games got worse. It wasn't learning, it was memorising noise and
calling it skill. If someone only ever tests on the same data they trained on, this is
what's happening under the hood.

**Leakage.** The pure cheat: sneak the answer into the question. I added one input, the
final goal difference of the match, and "predicted" the result. Accuracy: **100%.** Of
course it was. I told it the score. Every "100% accurate" predictor has an input like
this hiding in it, something that already contains the result: final possession,
post-match player ratings, the closing betting odds. When you see a perfect model, go
find which input is the answer wearing a fake moustache.

## What this was actually about

I never wanted a crystal ball. I wanted something honest enough that every guess comes
with a reason, and I wanted to find the exact line where the reasons run out.

That line turned out to be beautiful. I can call all 12 group winners, a clean 100%. I
can call 86% of knockout games one round ahead. But I cannot call a penalty shootout,
and I cannot break 64% on a single match, because past that line football stops being
about who's better and starts being about a ball hitting the inside of a post and
bouncing the wrong way.

A model that knows where its own line is, one that tells you "Spain 34%, here's exactly
why, and no I can't promise it," is worth ten models that shout a confident champion and
quietly leak the score.

---

*Everything here is reproducible and runs on nothing but Python, NumPy, and pandas.
Ratings from 49,503 international matches plus current national numbers; xG for all 94
World Cup matches from Sofascore. Code, data, and the full honest scoreboard are in the
repo.*
