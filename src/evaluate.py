"""Honest evaluation: walk-forward match accuracy, and World-Cup group-stage
qualification (who advances / who wins the group)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

import model as M

HOME, DRAW, AWAY = 0, 1, 2
HOSTS = {"United States", "Mexico", "Canada"}


def _auc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    c = sum((s > t) + 0.5 * (s == t) for s in pos for t in neg)
    return c / (len(pos) * len(neg))


def wdl_label(hs: int, as_: int) -> int:
    return HOME if hs > as_ else (AWAY if hs < as_ else DRAW)


def walk_forward(split: str = "2024-01-01", **fit_kw) -> dict:
    """Train on matches before `split`, test on the unseen ones after. This is the
    number that cannot be gamed by tuning on the World Cup."""
    r = M.load_results()
    tr = r[r["date"] < pd.Timestamp(split)]
    te = r[r["date"] >= pd.Timestamp(split)]
    mdl = M.fit_poisson(tr, **fit_kw)
    hits, ll = [], []
    for row in te.itertuples(index=False):
        p = mdl.wdl(row.home, row.away, neutral=bool(row.neutral_flag))
        y = wdl_label(row.hs, row.as_)
        hits.append(int(p.argmax() == y))
        ll.append(-math.log(max(float(p[y]), 1e-12)))
    return {"val_accuracy": float(np.mean(hits)), "val_logloss": float(np.mean(ll)),
            "val_matches": len(hits)}


def group_games() -> list[dict]:
    return [g for g in M.load_games() if str(g["type"]) == "group" and int(g["id"]) <= 72]


def actual_qualifiers() -> tuple[dict, dict]:
    """Return (group_winner_by_group, set_of_qualified_teams) from real results.
    Format: top 2 per group advance, plus the 8 best third-placed teams."""
    stand = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    groups = defaultdict(list)
    for g in group_games():
        if str(g.get("finished", "")).upper() != "TRUE":
            continue
        h, a = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        gr = g.get("group", "")
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        for t in (h, a):
            if t not in groups[gr]:
                groups[gr].append(t)
        stand[(gr, h)]["gf"] += hs; stand[(gr, h)]["ga"] += as_
        stand[(gr, a)]["gf"] += as_; stand[(gr, a)]["ga"] += hs
        if hs > as_:
            stand[(gr, h)]["pts"] += 3
        elif hs < as_:
            stand[(gr, a)]["pts"] += 3
        else:
            stand[(gr, h)]["pts"] += 1; stand[(gr, a)]["pts"] += 1
    return _rank(stand, groups)


def _rank(stand: dict, groups: dict) -> tuple[dict, set]:
    winners, ranked = {}, {}
    for gr, teams in groups.items():
        rows = sorted(teams, key=lambda t: (stand[(gr, t)]["pts"],
                      stand[(gr, t)]["gf"] - stand[(gr, t)]["ga"], stand[(gr, t)]["gf"]), reverse=True)
        winners[gr] = rows[0]
        ranked[gr] = rows
    qualified = set()
    thirds = []
    for gr, rows in ranked.items():
        qualified.update(rows[:2])
        if len(rows) >= 3:
            t = rows[2]
            thirds.append((stand[(gr, t)]["pts"], stand[(gr, t)]["gf"] - stand[(gr, t)]["ga"],
                           stand[(gr, t)]["gf"], gr, t))
    for *_, gr, t in sorted(thirds, reverse=True)[:8]:
        qualified.add(t)
    return winners, qualified


def model_qualifiers(mdl: M.PoissonModel) -> tuple[dict, set, dict]:
    """Expected-points standings from the model's W/D/L probabilities."""
    xpts = defaultdict(float)
    groups = defaultdict(list)
    match_hits, match_ll, draws_pred = [], [], 0
    for g in group_games():
        h, a = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        gr = g.get("group", "")
        for t in (h, a):
            if t not in groups[gr]:
                groups[gr].append(t)
        p = mdl.wdl(h, a, neutral=True, hosts=HOSTS)
        xpts[(gr, h)] += 3 * p[HOME] + p[DRAW]
        xpts[(gr, a)] += 3 * p[AWAY] + p[DRAW]
        if str(g.get("finished", "")).upper() == "TRUE":
            y = wdl_label(int(g["home_score"]), int(g["away_score"]))
            match_hits.append(int(p.argmax() == y))
            match_ll.append(-math.log(max(float(p[y]), 1e-12)))
            draws_pred += int(p.argmax() == DRAW)
    stand = {k: {"pts": v, "gf": 0, "ga": 0} for k, v in xpts.items()}  # rank by xpts only
    winners, ranked = {}, {}
    for gr, teams in groups.items():
        rows = sorted(teams, key=lambda t: xpts[(gr, t)], reverse=True)
        winners[gr] = rows[0]; ranked[gr] = rows
    qualified = set()
    thirds = []
    for gr, rows in ranked.items():
        qualified.update(rows[:2])
        if len(rows) >= 3:
            thirds.append((xpts[(gr, rows[2])], gr, rows[2]))
    for _, gr, t in sorted(thirds, reverse=True)[:8]:
        qualified.add(t)
    metrics = {"match_acc": float(np.mean(match_hits)), "match_logloss": float(np.mean(match_ll)),
               "match_n": len(match_hits), "draws_predicted": draws_pred}
    return winners, qualified, metrics


def ratings_qualification(host_bonus: float = 70.0) -> dict:
    """The headline model for WHO ADVANCES: rank each group by current national
    Elo, plus a home-advantage bonus for the three hosts (a real, ~70-Elo effect).
    No machine learning — and it beats the ML on this target."""
    elo = M.load_national_elo()
    groups = defaultdict(list)
    for g in group_games():
        gr = g.get("group", "")
        for t in (M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])):
            if t not in groups[gr]:
                groups[gr].append(t)

    def strength(t):
        return elo.get(t, 1500) + (host_bonus if t in HOSTS else 0)

    winners, ranked = {}, {}
    for gr, ts in groups.items():
        rows = sorted(ts, key=strength, reverse=True)
        winners[gr] = rows[0]; ranked[gr] = rows
    qualified, thirds = set(), []
    for gr, rows in ranked.items():
        qualified.update(rows[:2])
        if len(rows) >= 3:
            thirds.append((strength(rows[2]), gr, rows[2]))
    for _, gr, t in sorted(thirds, reverse=True)[:8]:
        qualified.add(t)

    aw, aq = actual_qualifiers()
    everyone = set().union(*ranked.values())
    return {"model": "national_elo+host", "group_winners": sum(1 for gr in aw if aw[gr] == winners.get(gr)),
            "groups": len(aw), "r32_hits": len(aq & qualified),
            "advance_auc": _auc([strength(t) for t in everyone], [t in aq for t in everyone]),
            "missed_winners": [(gr, aw[gr], winners[gr]) for gr in aw if aw[gr] != winners.get(gr)],
            "missed_qualifiers": sorted(aq - qualified)}


def qualification_scoreboard(mdl: M.PoissonModel) -> dict:
    aw, aq = actual_qualifiers()
    mw, mq, mm = model_qualifiers(mdl)
    winner_hits = sum(1 for gr in aw if aw[gr] == mw.get(gr))
    r32_hits = len(aq & mq)
    # per-team advancement classification accuracy over all 48 teams
    all_teams = set(aq) | set(mq)
    for gr in aw:
        all_teams |= set([aw[gr]])
    everyone = set()
    for g in group_games():
        everyone.add(M.canon(g["home_team_name_en"])); everyone.add(M.canon(g["away_team_name_en"]))
    tp = len(aq & mq); tn = len((everyone - aq) & (everyone - mq))
    advance_acc = (tp + tn) / len(everyone)
    # discrimination: does the model's expected-points rank the advancers above the rest?
    xp = defaultdict(float)
    for g in group_games():
        h, a = M.canon(g["home_team_name_en"]), M.canon(g["away_team_name_en"])
        gr = g.get("group", ""); p = mdl.wdl(h, a, neutral=True, hosts=HOSTS)
        xp[h] += 3 * p[HOME] + p[DRAW]; xp[a] += 3 * p[AWAY] + p[DRAW]
    advance_auc = _auc([xp[t] for t in everyone], [t in aq for t in everyone])
    return {**mm, "group_winners": winner_hits, "groups": len(aw),
            "r32_hits": r32_hits, "advance_acc": advance_acc,
            "advance_auc": advance_auc, "teams": len(everyone)}
