# How we taught a computer to predict the World Cup — and where it hit a wall

There's a fun question hiding inside "can a machine predict football." It's really
this: how much of a football match is skill, and how much is luck? Build an honest
predictor, push it as hard as it will go, and the exact spot where it stops
improving tells you where luck takes over. That turned out to be the best part of
the whole project. Let me walk you through what we built, in plain words, with the
real numbers.

## The one-sentence version of the model

For any two teams, ask one thing: who is better right now? Turn the answer into three
numbers, the chance team A wins, the chance of a draw, the chance team B wins, and
pick the biggest. That's it. Everything else in this post is us arguing about how to
measure "better," and being honest about when "better" still isn't enough to know who
wins.

## First, a wall we couldn't climb (and shouldn't have tried to)

We replayed all 72 group-stage matches, none of which the model had ever seen, and
asked it to call each one. It got **64%** right. We tried harder features, more
data, cleverer math. It stayed around 64%.

Here's why, and it's the whole point of the project. Of those 72 matches, **20
ended in a draw**. More than one in four. On those 20 draws, the model's average
guess for "draw" was only **17%**, while it was busy backing a winner at **66%**.
The model wasn't broken. Draws are just close to random. When two even teams play, a
draw is the most likely single result, yet the model almost never picks it, because
the two teams keep splitting the odds between them and neither "draw" pile ever wins.

Here's the number that settles it. Professional bookmakers, with billions of dollars
riding on getting this right, hit about **53–55%** on match outcomes. Our 64% is
genuinely good. So when someone tells you their model is 90% accurate at picking
match winners, they are either fooling themselves or feeding it the answer. I'll
show you exactly what that cheating looks like at the end.

So we stopped trying to win every match, and asked a better question.

## The question that *can* be answered: who advances?

Predicting one match is a coin weighted 60/40. But predicting who *wins a group* of
four teams — that's three matches, and luck starts to cancel out. A strong team
might draw one game and still top the group. This is a question where being right is
actually possible.

And here's the humbling part. We built a fancy machine-learning model. Then we tried
the dumbest thing imaginable: rank each group by one number, the team's current
strength rating, and call the top team the winner. **The dumb version got 11 of 12
group winners.** The fancy model didn't beat it. When your one-line baseline matches
your clever model, the baseline is the real story. For "who's the better team," a
good rating already knows almost everything.

The one group it got wrong was Group D. The model said Turkey. The **USA** won it.
Why did it miss? Because the rating couldn't see that the USA were playing at home.

## Home advantage isn't a light switch — it's a dimmer

Now, a World Cup has no "home team" in the usual sense — every match is at a neutral
stadium. (This tripped up our first model, which secretly gave a bonus to whichever
team was listed first in the fixture. That's not home advantage, that's alphabet
soup. We ripped it out.)

But 2026 is in the USA, Canada, and Mexico, so some teams are more at home than
others. Mexico playing in Guadalajara has 80,000 fans, no jet lag, and familiar
heat. Australia playing in New York has flown 15,000 km into a strange time zone. So
we made home advantage a dimmer instead of a switch. A team gets the full bonus in
its own country, and a slice of it that fades with the distance it had to travel.

The bonus, in rating points, came out about like you'd hope. Mexico got 80. The USA
and Canada got 95 at home. Haiti, Colombia, and Panama landed around 30 to 45, since
they're close by with big travelling crowds and a similar climate. Argentina got 5,
Japan 3, and Australia just 1, flown in from the far side of the planet. Add that
dimmer back in and the USA edge past Turkey. **Now the model calls all 12 group
winners. 100%.** The only thing between the rating and a perfect dozen was
remembering who's playing at home.

## Then the knockouts started — so we tried to predict them blind

Here's where it got genuinely exciting, because now we could test the model against
a future it truly hadn't seen. We froze its knowledge at the end of the group stage
and made it predict the entire knockout bracket — Round of 32 all the way to the
final — without peeking at a single knockout result.

Round of 32: **14 out of 16 correct.**

And look at the two it missed. Germany lost to Paraguay. The Netherlands lost to
Morocco. Both went to **penalty shootouts**. A shootout is a coin flip that happens
to involve feet, and no model on Earth should claim to call one. Of the 14 ties
decided in normal time by the better team, the model got all 14. That's not luck.
That's the ceiling.

## The upgrade you asked for: judge teams by *how they played*, not just the score

A scoreline lies. A team can play brilliantly and lose to one deflected goal in
stoppage time. Another can get outshot 20-to-3 and nick a 1-0. If you rate teams on
results alone, you're rating them partly on luck. So we fed the model three richer
signals.

**Expected goals (xG).** This is the big one. xG asks: given the quality of the
chances a team created, how many goals *should* they have scored? It's the closest
thing football has to measuring who deserved to win. We pulled the real xG for all
94 matches.

It immediately told us things the scoreboard hid. Egypt finished the group stage
with a +2 goal difference, which looks good, but their xG difference was **−0.8**.
The chances they gave up were worse than the chances they made. They'd been a little
lucky. Norway were the opposite. Their results looked ordinary, but their xG was
strong, so they were better than they looked. And here's the payoff. When Norway
later knocked Brazil out, the xG wasn't surprised. It had been quietly saying Norway
were underrated the whole time.

**The manner of the result.** Losing to a 93rd-minute winner means you were level
for 92 minutes. You played well. So we read the goal minutes and gave a team that
only lost in the dying seconds most of the credit of a draw. Getting beaten late and
getting battered should not count the same.

**Momentum.** A team that loses its opener but wins games two and three is rounding
into form, and that's more dangerous than a team that started hot and faded. So the
third group game counts for more than the first.

**Squad quality.** This is the "how good are these players, really" signal, and it
doesn't care about one noisy group. We measured it from the players' club form. A
team built from top clubs is strong even if it stumbled for three games.

Fold all of that in and the blind Round of 32 climbs to **14 of 16**. The only two
misses, you guessed it, were the penalty shootouts. Egypt beating Australia flipped
to correct, because Egypt's players come from better clubs than their raw rating
knew. The quality signals paid for themselves.

## Rolling it forward, one round at a time

As the tournament moved on, we moved the cutoff with it. Use everything through the
Round of 32; predict the Round of 16 blind. Then use everything through the Round of
16; predict the rest.

- Groups → Round of 32: **14/16**
- Round of 32 → Round of 16: **5/6**
- **Total, predicting each round from the one before: 19 of 22 — 86%.**

The three misses across the whole knockout stage were two penalty shootouts and one
honest on-the-day upset, Norway beating Brazil in normal time. And here's a small win
for the xG idea. Morocco actually outplayed the Netherlands on chances, 1.38 xG to
0.24, even while scraping through on penalties. So the model rated Morocco up, and
then correctly picked them to beat Canada in the next round.

## Where it stands right now

We take everything that has actually happened, every group game, the Round of 32, and
the finished Round of 16, and simulate the rest of the bracket 20,000 times. The
title is Spain's to lose. Here is the full bracket, with the model's predicted path
through the games still to be played:

![Predicted World Cup 2026 bracket](outputs/predicted_bracket.svg)

| Team | Chance of winning it all |
|---|---:|
| **Spain** | **34%** |
| France | 20% |
| Argentina | 18% |
| England | 13% |
| Norway | 5% |

Spain sit clear at the top because their underlying numbers — not just their results,
their *xG* — were the best in the tournament. The single most likely bracket has
**Spain beating Argentina in the final.**

## The two ways to fake a perfect model (so you can spot them)

We promised to show you the cheating. There are two tricks, and we ran both on our
own model so you'd recognize them in the wild.

**Overfitting.** Give a model far more knobs than it needs and let it memorize the
training data. Its accuracy on matches it has already seen climbs toward 100% — while
its accuracy on *new* matches quietly gets worse. It's not learning; it's memorizing
noise and calling it skill. If someone only ever tests their model on the same data
they trained it on, this is what's happening.

**Leakage.** The purest cheat is to sneak the answer into the question. We added one
input, the final goal difference of the match, and "predicted" the result. Accuracy:
**100%.** Of course it was. We told it the score. Every "100% accurate" predictor has
an input like this hiding in it, something that already contains the result: final
possession, post-match player ratings, the closing betting odds. When you see a
perfect model, go find which input is the answer wearing a fake moustache.

## What this was really about

The point was never a crystal ball. It was to build something honest enough that
every guess comes with a reason, and to find the exact line where reasons run out.

That line is a beautiful thing to find. We can call all 12 group winners, a perfect
100%. We can predict 86% of knockout games one round ahead. But we cannot call a
penalty shootout, and we cannot break 64% on a single match, because past that line
football stops being about who's better and starts being about a ball hitting the
inside of a post and bouncing out. A model that knows where its own line is, one that
says "Spain 34%, here's why, and no I can't promise it," is worth ten models that
shout a confident winner and quietly leak the score.

---

*Everything here is reproducible. Ratings come from decades of international results
plus current strength numbers; xG for all 94 World Cup matches came from Sofascore.
The code runs on nothing but Python, NumPy, and pandas. Repo, data, and the honest
scoreboard are all in the project folder.*
