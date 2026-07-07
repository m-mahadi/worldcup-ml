"""Predict the knockout bracket R32 -> Final, BLIND.

Data cutoff = end of the group stage. We take the actual group-stage results
(now known), update every team's rating with them, and then predict the entire
knockout tree without ever looking at a knockout result. The R32 match-ups are
fixed by the final group standings, and the slot-tree (which winner meets which)
is structural — both are group-stage information, not results.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import model as M
import geo
import evaluate as E
import quality as Q

# Verified bracket tree (child game id -> the two parent game ids whose winners meet).
# Reconstructed from the fixed group-seeded structure, not from who won.
BRACKET = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),   # R16 = R32 winners
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),   # QF  = R16 winners
    101: (97, 98), 102: (99, 100),                             # SF  = QF winners
    104: (101, 102),                                           # Final = SF winners
}
ROUND_NAME = {"r32": "Round of 32", "r16": "Round of 16", "qf": "Quarter-final",
              "sf": "Semi-final", "final": "Final"}


def games_by_id() -> dict:
    return {int(g["id"]): g for g in M.load_games()}


# How much a group result counts by matchday. Later games count more: a team that
# rounds into form (loses matchday 1, wins 2 and 3) should rise, not be dragged
# down by a bad opener. Difficulty of the group is handled automatically by Elo —
# drawing Brazil moves you far more than beating a minnow.
MATCHDAY_WEIGHT = {1: 0.6, 2: 1.0, 3: 1.6}
GROUP_K = 80.0  # base weight for a matchday-2 result; the only in-tournament signal


def post_group_elo(base_k: float = GROUP_K, squad_w: float = 25.0) -> dict[str, float]:
    """Post-group team ratings, using every quality signal we have:
      - starting point: current national Elo (all nations, results-based)
      - group results graded by *performance*, not just W/D/L: xG-deserved result
        (luck stripped out) blended with the manner-adjusted score
      - opponent difficulty: automatic in Elo's expected score
      - momentum: later matchdays weighted more (rounding into form)
      - squad quality: a small nudge from xG-based SPI club ratings
    This is the 'use the group stage' step — the only new info after the cutoff."""
    elo = dict(M.load_national_elo())
    xg = Q.load_xg()
    group = [g for g in M.load_games()
             if str(g["type"]) == "group" and str(g.get("finished", "")).upper() == "TRUE"]
    group.sort(key=lambda g: int(g.get("matchday", 0)))  # sequential form, MD1 -> MD3
    for g in group:
        h, a = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        md_w = MATCHDAY_WEIGHT.get(int(g.get("matchday", 2)), 1.0)  # momentum
        ea = 1 / (1 + 10 ** ((elo.get(a, 1500) - elo.get(h, 1500)) / 400))  # opponent difficulty
        perf_h, _ = Q.match_performance(g, xg)  # xG-deserved + manner, in [0,1]
        gd = abs(int(g["home_score"]) - int(g["away_score"]))
        k = base_k * md_w * (1.0 if gd <= 1 else 1.25 if gd == 2 else 1.5)  # mild margin
        delta = k * (perf_h - ea)
        elo[h] = elo.get(h, 1500) + delta
        elo[a] = elo.get(a, 1500) - delta
    # small static prior: xG-based squad quality (helps thinly-rated nations)
    if squad_w:
        for team, z in Q.squad_quality().items():
            elo[M.canon(team)] = elo.get(M.canon(team), 1500) + squad_w * z
    return elo


def apply_knockout_round(elo: dict, stage_ids: range, base_k: float = GROUP_K) -> dict:
    """Update ratings with completed knockout matches, graded by performance
    (xG-deserved + manner), so a penalty-shootout 'winner' who was outplayed does
    NOT rise. Used to move the data cutoff forward one round at a time."""
    gid = games_by_id()
    xg = Q.load_xg()
    for i in stage_ids:
        g = gid.get(i)
        if not g or str(g.get("finished", "")).upper() != "TRUE":
            continue
        h, a = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        ea = 1 / (1 + 10 ** ((elo.get(a, 1500) - elo.get(h, 1500)) / 400))
        perf_h, _ = Q.match_performance(g, xg)  # shootouts read ~0.5/0.5 unless xG says otherwise
        gd = abs(int(g["home_score"]) - int(g["away_score"]))
        k = base_k * (1.0 if gd <= 1 else 1.25 if gd == 2 else 1.5)
        delta = k * (perf_h - ea)
        elo[h] = elo.get(h, 1500) + delta
        elo[a] = elo.get(a, 1500) - delta
    return elo


def post_r32_elo() -> dict[str, float]:
    """Ratings after the group stage AND the Round of 32 — the cutoff for
    predicting the Round of 16."""
    return apply_knockout_round(post_group_elo(), range(73, 89))


def post_r16_elo() -> dict[str, float]:
    """Ratings after everything played so far: groups + R32 + the finished R16
    games. apply_knockout_round skips unfinished matches, so pending R16 ties
    (95, 96) are left to be predicted."""
    return apply_knockout_round(post_r32_elo(), range(89, 97))


def predict_forward(mdl: M.PoissonModel, deterministic=True, rng=None) -> dict[int, str]:
    """Use ALL results so far as facts and predict only the unplayed games.
    Every finished match keeps its real advancer; every unplayed match is
    predicted from whoever (real or predicted) reaches it."""
    gid = games_by_id()
    winners: dict[int, str] = {}

    def decide(t1, t2, stadium):
        pa = p_advance(mdl, t1, t2, stadium)
        if deterministic:
            return t1 if pa >= 0.5 else t2
        return t1 if rng.random() < pa else t2

    order = list(range(73, 89)) + list(range(89, 97)) + [97, 98, 99, 100, 101, 102, 104, 103]
    for i in order:
        g = gid[i]
        if i <= 96:                       # R32 + R16 have real match-ups in the fixture
            t1 = M.canon(g.get("home_team_name_en") or "")
            t2 = M.canon(g.get("away_team_name_en") or "")
        elif i == 103:                    # third place = the two semi losers
            t1 = winners[97] if winners[101] != winners[97] else winners[98]
            t2 = winners[99] if winners[102] != winners[99] else winners[100]
        else:                             # QF, SF, Final from the bracket
            pa, pb = BRACKET[i]
            t1, t2 = winners[pa], winners[pb]
        if str(g.get("finished", "")).upper() == "TRUE":
            winners[i] = actual_advancer(gid, i)      # a fact, not a prediction
        else:
            winners[i] = decide(t1, t2, g["stadium_id"])
    return winners


def champion_probabilities_forward(mdl: M.PoissonModel, runs: int = 20000) -> list[dict]:
    """Champion odds conditioned on everything that has actually happened: finished
    games are fixed, only the remaining bracket is simulated."""
    rng = random.Random(2026)
    champ, final, semi = Counter(), Counter(), Counter()
    for _ in range(runs):
        w = predict_forward(mdl, deterministic=False, rng=rng)
        champ[w[104]] += 1
        final[w[101]] += 1; final[w[102]] += 1
        for i in (97, 98, 99, 100):
            semi[w[i]] += 1
    teams = set(champ) | set(final) | set(semi)
    rows = [{"team": t, "champion_%": 100 * champ[t] / runs, "final_%": 100 * final[t] / runs,
             "semifinal_%": 100 * semi[t] / runs} for t in teams]
    return sorted(rows, key=lambda r: -r["champion_%"])


def build_model(elo: dict) -> M.PoissonModel:
    mdl = M.fit_poisson(M.load_results(), elo=elo, elo_blend=0.5, half_life_years=2.5)
    return mdl


def p_advance(mdl: M.PoissonModel, t1: str, t2: str, stadium_id) -> float:
    """Probability t1 knocks t2 out. No draws in knockouts: a level game is
    resolved (extra time / penalties) roughly in proportion to strength."""
    sid2city = E.geo._stadium_city()
    p = mdl.wdl(t1, t2, adv1=E._geo_adv(mdl, t1, stadium_id, sid2city),
                adv2=E._geo_adv(mdl, t2, stadium_id, sid2city))
    win, draw, lose = float(p[0]), float(p[1]), float(p[2])
    denom = win + lose
    return win + draw * (win / denom if denom > 0 else 0.5)


def r32_matchups(gid: dict) -> list[tuple[int, str, str, str]]:
    return [(i, M.canon(gid[i]["home_team_name_en"]), M.canon(gid[i]["away_team_name_en"]),
             gid[i]["stadium_id"]) for i in range(73, 89)]


def predict_bracket(mdl: M.PoissonModel, deterministic=True, rng=None, start="r32") -> dict[int, str]:
    """Play the tree forward from `start` round. start='r32' predicts everything
    from the group cutoff; start='r16' seeds the ACTUAL Round-of-16 match-ups
    (known once the R32 is complete) and predicts from there."""
    gid = games_by_id()
    winners: dict[int, str] = {}

    def play(t1, t2, stadium):
        pa = p_advance(mdl, t1, t2, stadium)
        if deterministic:
            return t1 if pa >= 0.5 else t2
        return t1 if rng.random() < pa else t2

    if start == "r32":
        for i, t1, t2, stad in r32_matchups(gid):
            winners[i] = play(t1, t2, stad)
        r16_pair = lambda i: (winners[BRACKET[i][0]], winners[BRACKET[i][1]])
    else:  # start == "r16": use the real R16 match-ups
        r16_pair = lambda i: (M.canon(gid[i]["home_team_name_en"]), M.canon(gid[i]["away_team_name_en"]))

    for i in range(89, 97):                           # R16
        t1, t2 = r16_pair(i)
        winners[i] = play(t1, t2, gid[i]["stadium_id"])
    for i in (97, 98, 99, 100, 101, 102, 104):        # QF -> Final
        pa, pb = BRACKET[i]
        winners[i] = play(winners[pa], winners[pb], gid[i]["stadium_id"])
    l1 = [t for t in (winners[97], winners[98]) if t != winners[101]][0]
    l2 = [t for t in (winners[99], winners[100]) if t != winners[102]][0]
    winners[103] = play(l1, l2, gid[103]["stadium_id"])
    return winners


def round_backtest(mdl: M.PoissonModel, game_ids: range) -> dict:
    """Blind accuracy on the actual match-ups of one knockout round: predict each,
    then reveal the real advancer. Uses results only to score, never to predict."""
    gid = games_by_id()
    hits, rows = [], []
    for i in game_ids:
        g = gid.get(i)
        if not g or str(g.get("finished", "")).upper() != "TRUE":
            continue
        t1, t2 = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        pick = t1 if p_advance(mdl, t1, t2, g["stadium_id"]) >= 0.5 else t2
        actual = actual_advancer(gid, i)
        if actual is not None:
            hits.append(int(pick == actual))
            rows.append((i, t1, t2, pick, actual, int(pick == actual)))
    return {"accuracy": sum(hits) / len(hits) if hits else float("nan"), "n": len(hits), "rows": rows}


def champion_probabilities(mdl: M.PoissonModel, runs: int = 20000, start: str = "r32") -> list[dict]:
    rng = random.Random(2026)
    champ, final, semi = Counter(), Counter(), Counter()
    for _ in range(runs):
        w = predict_bracket(mdl, deterministic=False, rng=rng, start=start)
        champ[w[104]] += 1
        final[w[101]] += 1; final[w[102]] += 1
        for i in (97, 98, 99, 100):
            semi[w[i]] += 1
    teams = set(champ) | set(final) | set(semi)
    rows = [{"team": t, "champion_%": 100 * champ[t] / runs, "final_%": 100 * final[t] / runs,
             "semifinal_%": 100 * semi[t] / runs} for t in teams]
    return sorted(rows, key=lambda r: -r["champion_%"])


def _child_of() -> dict[int, int]:
    child = {}
    for c, (pa, pb) in BRACKET.items():
        child[pa] = c; child[pb] = c
    return child


def actual_advancer(gid: dict, i: int) -> str | None:
    """The team that really advanced from game i — read from who appears in the
    next round (so penalty-shootout winners are handled correctly)."""
    g = gid[i]
    if str(g.get("finished", "")).upper() != "TRUE":
        return None
    teams = {M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])}
    ch = _child_of().get(i)
    if ch is not None:
        cg = gid[ch]
        cteams = {M.canon(cg.get("home_team_name_en") or ""), M.canon(cg.get("away_team_name_en") or "")}
        hit = teams & cteams
        if hit:
            return hit.pop()
    hs, as_ = int(g["home_score"]), int(g["away_score"])   # fallback: decisive game
    return M.canon(g["home_team_name_en"]) if hs > as_ else M.canon(g["away_team_name_en"])


def r32_backtest(mdl: M.PoissonModel) -> dict:
    """Blind R32 accuracy: predict each fixed R32 match, then reveal the real
    advancer. Uses actual results ONLY to score, never to predict."""
    gid = games_by_id()
    hits, rows = [], []
    for i, t1, t2, stad in r32_matchups(gid):
        pick = t1 if p_advance(mdl, t1, t2, stad) >= 0.5 else t2
        actual = actual_advancer(gid, i)
        if actual is not None:
            hits.append(int(pick == actual))
            rows.append((i, t1, t2, pick, actual, int(pick == actual)))
    return {"accuracy": sum(hits) / len(hits), "n": len(hits), "rows": rows}
