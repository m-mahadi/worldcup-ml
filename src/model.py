"""Goals-based (Poisson) World Cup model — pure numpy/pandas.

Instead of guessing win/draw/loss directly, we model how many goals each team is
expected to score. That produces *real* draw probabilities and lets us simulate
group tables. Team attack/defence strengths are fit by weighted maximum
likelihood on 49k historical internationals, weighting recent and important
matches more. A current national-Elo prior is optionally blended in for teams we
have little recent data on.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WC_START = pd.Timestamp("2026-06-11")

ALIASES = {
    "Democratic Republic of the Congo": "DR Congo", "Congo DR": "DR Congo",
    "Côte D'Ivoire": "Ivory Coast", "Czechia": "Czech Republic",
    "Korea Republic": "South Korea", "IR Iran": "Iran", "USA": "United States",
    "Chinese Taipei": "Taiwan", "Cape Verde Islands": "Cape Verde",
}


def canon(name: object) -> str:
    s = re.sub(r"\s+", " ", "" if name is None else str(name).strip())
    return ALIASES.get(s, s)


def importance(tournament: str) -> float:
    t = str(tournament).lower()
    is_qual = "qualification" in t or "qualifier" in t
    if "world cup" in t and not is_qual:
        return 1.8
    if any(x in t for x in ["uefa euro", "copa am", "african cup", "asian cup", "gold cup", "confederations"]) and not is_qual:
        return 1.5
    if "nations league" in t or is_qual:
        return 1.2
    if "friendly" in t:
        return 0.5
    return 0.9


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_results(before: pd.Timestamp | None = WC_START) -> pd.DataFrame:
    r = pd.read_csv(RAW / "international_results.csv")
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r = r.dropna(subset=["date", "home_score", "away_score"])
    if before is not None:
        r = r[r["date"] < before]
    r = r.sort_values("date").reset_index(drop=True)
    r["home"] = r["home_team"].map(canon)
    r["away"] = r["away_team"].map(canon)
    r["hs"] = r["home_score"].astype(int)
    r["as_"] = r["away_score"].astype(int)
    r["neutral_flag"] = r["neutral"].astype(str).str.upper().eq("TRUE")
    r["imp"] = r["tournament"].map(importance)
    return r


def load_national_elo() -> dict[str, float]:
    code2name = {}
    for parts in csv.reader((RAW / "eloratings_teams.tsv").open(encoding="utf-8", errors="replace"), delimiter="\t"):
        if len(parts) >= 2:
            code2name[parts[0]] = parts[1]
    elo = {}
    for parts in csv.reader((RAW / "eloratings_world.tsv").open(encoding="utf-8", errors="replace"), delimiter="\t"):
        if len(parts) >= 4 and parts[2] in code2name:
            try:
                elo[canon(code2name[parts[2]])] = float(parts[3])
            except ValueError:
                pass
    return elo


def load_squad_spi_strength() -> dict[str, float]:
    """Average SPI rating of each squad's clubs (covers 25 leagues incl. MLS,
    Saudi, Turkey — the non-Big-Five gap). Returns a z-scored per-team value."""
    squad = pd.read_csv(RAW / "squad_players.csv")
    spi = pd.read_csv(RAW / "spi_club_rankings.csv")

    def norm(s):
        s = str(s).encode("latin1", "ignore").decode("utf-8", "ignore")  # fix mojibake
        s = re.sub(r"\(.*?\)", "", s.lower())
        s = re.sub(r"\b(fc|sc|cf|sk|ac|afc|club|de|city|united)\b", "", s)
        return re.sub(r"[^a-z0-9]", "", s)

    spi_map = {norm(n): v for n, v in zip(spi["name"], spi["spi"])}
    squad["cn"] = squad["club"].map(norm)
    squad["club_spi"] = squad["cn"].map(spi_map)
    team_spi = squad.groupby(squad["team"].map(canon))["club_spi"].mean()
    z = (team_spi - team_spi.mean()) / team_spi.std()
    return z.fillna(0.0).to_dict()


def load_games() -> list[dict]:
    games = json.loads((RAW / "worldcup2026_api_games.json").read_text(encoding="utf-8"))["games"]
    return sorted(games, key=lambda g: int(g["id"]))


# ---------------------------------------------------------------------------
# Poisson attack/defence model
# ---------------------------------------------------------------------------
@dataclass
class PoissonModel:
    teams: dict[str, int]
    att: np.ndarray
    dfn: np.ndarray
    mu: float
    gamma: float          # home advantage (applied only on non-neutral games)
    elo: dict[str, float]
    elo_blend: float      # 0 = ignore national Elo, >0 = pull strength toward it
    max_goals: int = 10

    def _rate(self, team: str, opp: str, home_adv: float) -> float:
        i, j = self.teams.get(team), self.teams.get(opp)
        a_i = self.att[i] if i is not None else 0.0
        d_j = self.dfn[j] if j is not None else 0.0
        lam = math.exp(self.mu + a_i - d_j + home_adv)
        if self.elo_blend > 0:
            # nudge by national-Elo strength difference (covers thin-data teams)
            e = (self.elo.get(team, 1500) - self.elo.get(opp, 1500)) / 400.0
            lam *= math.exp(self.elo_blend * e)
        return lam

    def score_matrix(self, t1: str, t2: str, adv1: float = 0.0, adv2: float = 0.0) -> np.ndarray:
        # adv1/adv2 are log-goal home-advantage terms (0 = neutral). At a World Cup
        # only the three hosts get it; everyone else plays truly neutral.
        l1 = self._rate(t1, t2, adv1)
        l2 = self._rate(t2, t1, adv2)
        k = np.arange(self.max_goals + 1)
        logf = np.array([math.lgamma(x + 1) for x in k])
        p1 = np.exp(-l1 + k * math.log(l1) - logf)
        p2 = np.exp(-l2 + k * math.log(l2) - logf)
        return np.outer(p1, p2)  # M[x,y] = P(t1 scores x, t2 scores y)

    def wdl(self, t1: str, t2: str, neutral: bool = True, hosts: set | None = None) -> np.ndarray:
        if hosts:  # World Cup: only a host nation gets the fitted home advantage
            adv1 = self.gamma if t1 in hosts else 0.0
            adv2 = self.gamma if t2 in hosts else 0.0
        else:      # historical: t1 is the home side unless the match was neutral
            adv1 = 0.0 if neutral else self.gamma
            adv2 = 0.0
        M = self.score_matrix(t1, t2, adv1, adv2)
        p_t1 = np.tril(M, -1).sum()   # t1 goals > t2 goals
        p_draw = np.trace(M)
        p_t2 = np.triu(M, 1).sum()
        s = p_t1 + p_draw + p_t2
        return np.array([p_t1 / s, p_draw / s, p_t2 / s])


def fit_poisson(r: pd.DataFrame, half_life_years: float = 2.5, l2: float = 0.02,
                iters: int = 400, lr: float = 0.05, elo: dict | None = None,
                elo_blend: float = 0.0) -> PoissonModel:
    teams = {t: i for i, t in enumerate(sorted(set(r["home"]) | set(r["away"])))}
    T = len(teams)
    h = r["home"].map(teams).to_numpy()
    a = r["away"].map(teams).to_numpy()
    gh = r["hs"].to_numpy(dtype=float)
    ga = r["as_"].to_numpy(dtype=float)
    home_flag = (~r["neutral_flag"]).to_numpy(dtype=float)

    age_days = (WC_START - r["date"]).dt.days.to_numpy(dtype=float)
    w = r["imp"].to_numpy(dtype=float) * 0.5 ** (age_days / (365.0 * half_life_years))
    w /= w.mean()

    att = np.zeros(T)
    dfn = np.zeros(T)
    mu = math.log(max(np.average((gh + ga) / 2, weights=w), 0.3))
    gamma = 0.25
    for _ in range(iters):
        lam_h = np.exp(mu + att[h] - dfn[a] + gamma * home_flag)
        lam_a = np.exp(mu + att[a] - dfn[h])
        rh = w * (lam_h - gh)   # dNLL/dλ_h * λ_h
        ra = w * (lam_a - ga)
        g_att = np.zeros(T); g_dfn = np.zeros(T)
        np.add.at(g_att, h, rh); np.add.at(g_att, a, ra)
        np.add.at(g_dfn, a, -rh); np.add.at(g_dfn, h, -ra)
        g_att = g_att / len(r) + l2 * att
        g_dfn = g_dfn / len(r) + l2 * dfn
        g_mu = (rh + ra).sum() / len(r)
        g_gamma = (rh * home_flag).sum() / len(r)
        att -= lr * g_att; dfn -= lr * g_dfn
        mu -= lr * g_mu; gamma -= lr * g_gamma
        att -= att.mean(); dfn -= dfn.mean()   # identifiability
    return PoissonModel(teams, att, dfn, mu, gamma, elo or {}, elo_blend)
