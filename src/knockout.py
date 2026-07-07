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


def post_group_elo() -> dict[str, float]:
    """Start from current national Elo and apply Elo updates from the 72 actual
    group matches. Teams that over-performed in the group stage (Morocco, Norway)
    rise; flat-track bullies who stumbled fall. This is the 'use the group results'
    step — the only new information after the cutoff."""
    elo = dict(M.load_national_elo())
    for g in M.load_games():
        if str(g["type"]) != "group" or str(g.get("finished", "")).upper() != "TRUE":
            continue
        h, a = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        ea = 1 / (1 + 10 ** ((elo.get(a, 1500) - elo.get(h, 1500)) / 400))
        sa = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        k = 40 * (1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8)  # WC weight + margin
        delta = k * (sa - ea)
        elo[h] = elo.get(h, 1500) + delta
        elo[a] = elo.get(a, 1500) - delta
    return elo


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


def predict_bracket(mdl: M.PoissonModel, deterministic=True, rng=None) -> dict[int, str]:
    """Play the whole tree forward. Returns winner of each game id."""
    gid = games_by_id()
    winners: dict[int, str] = {}

    def play(gid_num, t1, t2, stadium):
        pa = p_advance(mdl, t1, t2, stadium)
        if deterministic:
            return t1 if pa >= 0.5 else t2
        return t1 if rng.random() < pa else t2

    for i, t1, t2, stad in r32_matchups(gid):        # R32
        winners[i] = play(i, t1, t2, stad)
    for i in range(89, 105):                          # R16 -> Final
        if i not in BRACKET:
            continue
        pa, pb = BRACKET[i]
        winners[i] = play(i, winners[pa], winners[pb], gid[i]["stadium_id"])
    # third place (losers of the semis)
    l1 = [t for t in (winners[BRACKET[101][0]], winners[BRACKET[101][1]]) if t != winners[101]][0]
    l2 = [t for t in (winners[BRACKET[102][0]], winners[BRACKET[102][1]]) if t != winners[102]][0]
    winners[103] = play(103, l1, l2, gid[103]["stadium_id"])
    return winners


def champion_probabilities(mdl: M.PoissonModel, runs: int = 20000) -> list[dict]:
    rng = random.Random(2026)
    champ, final, semi = Counter(), Counter(), Counter()
    for _ in range(runs):
        w = predict_bracket(mdl, deterministic=False, rng=rng)
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
