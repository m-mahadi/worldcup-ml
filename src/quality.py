"""Quality-of-team signals that go beyond win/draw/loss.

Football results are noisy: a team can play well and lose to a late goal or a
penalty shootout, or win ugly while being outplayed. We fold in the *manner* of
each group result and each squad's underlying club quality, so the ratings we
carry into the knockouts reflect how good teams actually are — not just what the
scoreboard said.

Signals used (all from data we hold, no fragile live scraping):
  - manner of result: the scoreline at 85' blended with full-time, so a team that
    only lost to a stoppage-time goal keeps most of its credit.
  - margin / dominance: bigger, earlier leads count for more (via Elo K).
  - opponent difficulty: automatic in Elo (drawing Brazil >> beating a minnow).
  - momentum: later matchdays weighted more (rounding into form).
  - squad quality: SPI club ratings are xG-derived; a squad of xG-strong clubs is
    a strong squad regardless of one noisy group.
"""
from __future__ import annotations

import re

import model as M


def parse_minutes(s: object) -> list[int]:
    """Pull goal minutes from a scorer string like {"R. Jimenez 67'","x 90+3'"}."""
    if not s or str(s) == "null":
        return []
    return [int(m) + (int(x) if x else 0) for m, x in re.findall(r"(\d+)(?:\+(\d+))?'", str(s))]


def _res(a: float, b: float) -> float:
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def performance_result(game: dict, cutoff: int = 85, blend: float = 0.35) -> tuple[float, float]:
    """Home team's manner-adjusted 'how well did they do' score in [0,1] (and the
    away mirror). Full-time result blended with the scoreline at `cutoff` minutes,
    so late swings are partly discounted — losing to a 90+' goal ~ a draw's worth."""
    hs, as_ = int(game["home_score"]), int(game["away_score"])
    hm, am = parse_minutes(game.get("home_scorers")), parse_minutes(game.get("away_scorers"))
    h_cut = sum(1 for m in hm if m <= cutoff)
    a_cut = sum(1 for m in am if m <= cutoff)
    perf_home = (1 - blend) * _res(hs, as_) + blend * _res(h_cut, a_cut)
    return perf_home, 1.0 - perf_home


def load_xg() -> dict[int, tuple[float, float]]:
    """match_id -> (team1_xg, team2_xg), aligned to the fixture's team1/team2 order."""
    import pandas as pd
    df = pd.read_csv(M.RAW / "match_xg_stats.csv")
    out = {}
    for _, r in df.iterrows():
        try:
            out[int(r["match_id"])] = (float(r["team1_xg"]), float(r["team2_xg"]))
        except (ValueError, TypeError):
            pass
    return out


def _poisson_pmf(k: int, lam: float) -> float:
    import math
    return math.exp(-lam + k * math.log(max(lam, 1e-9)) - math.lgamma(k + 1))


def xg_deserved(xg1: float, xg2: float, max_goals: int = 8) -> float:
    """'Expected points share' team1 deserved from its xG: treat each side's xG as
    a Poisson mean and compute P(win) + 0.5*P(draw). This is who *deserved* to win
    on the balance of chances — the honest read of a match, luck stripped out."""
    p1 = [_poisson_pmf(k, xg1) for k in range(max_goals + 1)]
    p2 = [_poisson_pmf(k, xg2) for k in range(max_goals + 1)]
    win = draw = 0.0
    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            if a > b:
                win += p1[a] * p2[b]
            elif a == b:
                draw += p1[a] * p2[b]
    return win + 0.5 * draw


def match_performance(game: dict, xg: dict, w_xg: float = 0.55) -> tuple[float, float]:
    """Team1's overall performance score in [0,1], blending the xG-deserved result
    with the manner-adjusted actual result. Falls back to manner-only if no xG."""
    perf_manner, _ = performance_result(game)
    mid = int(game["id"])
    if mid in xg:
        xg1, xg2 = xg[mid]
        perf = w_xg * xg_deserved(xg1, xg2) + (1 - w_xg) * perf_manner
    else:
        perf = perf_manner
    return perf, 1.0 - perf


def squad_quality() -> dict[str, float]:
    """Per-team xG-based squad quality: average SPI rating of each player's club
    (SPI off/def are xG-derived). Covers 25 leagues incl. MLS/Saudi/Turkey.
    Returned z-scored across the World Cup field; 0.0 where we can't see a club."""
    import numpy as np
    import pandas as pd
    squad = pd.read_csv(M.RAW / "squad_players.csv")
    spi = pd.read_csv(M.RAW / "spi_club_rankings.csv")

    def norm(s):
        s = str(s).encode("latin1", "ignore").decode("utf-8", "ignore")
        s = re.sub(r"\(.*?\)", "", s.lower())
        s = re.sub(r"\b(fc|sc|cf|sk|ac|afc|club|de|city|united|calcio)\b", "", s)
        return re.sub(r"[^a-z0-9]", "", s)

    spi_map = {norm(n): v for n, v in zip(spi["name"], spi["spi"])}
    squad["q"] = squad["club"].map(norm).map(spi_map)
    team_q = squad.groupby(squad["team"].map(M.canon))["q"].mean()
    z = (team_q - team_q.mean()) / team_q.std()
    return z.reindex(z.index).fillna(0.0).to_dict()
