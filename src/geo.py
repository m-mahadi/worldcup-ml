"""Graded home advantage by geography.

A neutral venue isn't equally neutral for everyone. A team playing near home
travels less, keeps its climate/timezone, and brings far more fans. We model this
as a home-advantage bonus that decays smoothly with the distance from a team's
country to the actual match venue — so the three hosts get the most, CONCACAF and
South American sides get a real slice, and teams flown in from Europe/Africa/Asia/
Oceania get almost nothing. "Host advantage" is just the distance-0 special case.
"""
from __future__ import annotations

import math

import model as M

HOME_ELO_MAX = 95.0   # Elo-equivalent of a true home game (home adv ~ 60-100 Elo)
DECAY_KM = 3000.0     # distance at which the bonus falls to 1/e of its peak

# 16 host cities (lat, lon)
CITY = {
    "Mexico City": (19.43, -99.13), "Guadalajara": (20.67, -103.35), "Monterrey": (25.67, -100.32),
    "Dallas": (32.78, -96.80), "Houston": (29.76, -95.37), "Kansas City": (39.10, -94.58),
    "Atlanta": (33.75, -84.39), "Miami": (25.76, -80.19), "Boston": (42.36, -71.06),
    "Philadelphia": (39.95, -75.17), "New York/New Jersey": (40.71, -74.01), "Toronto": (43.65, -79.38),
    "Vancouver": (49.28, -123.12), "Seattle": (47.61, -122.33), "San Francisco Bay Area": (37.37, -121.97),
    "Los Angeles": (34.05, -118.24),
}

# WC-2026 nations -> approximate country centroid (lat, lon)
TEAM = {
    "Algeria": (28.0, 3.0), "Argentina": (-38.4, -63.6), "Australia": (-25.3, 133.8),
    "Austria": (47.5, 14.5), "Belgium": (50.5, 4.5), "Bosnia and Herzegovina": (43.9, 17.7),
    "Brazil": (-14.2, -51.9), "Canada": (56.1, -106.3), "Cape Verde": (16.0, -24.0),
    "Colombia": (4.6, -74.3), "Croatia": (45.1, 15.2), "Curaçao": (12.2, -69.0),
    "Czech Republic": (49.8, 15.5), "DR Congo": (-4.0, 21.8), "Ecuador": (-1.8, -78.2),
    "Egypt": (26.8, 30.8), "England": (52.4, -1.5), "France": (46.6, 2.2), "Germany": (51.2, 10.4),
    "Ghana": (7.9, -1.0), "Haiti": (19.0, -72.3), "Iran": (32.4, 53.7), "Iraq": (33.2, 43.7),
    "Ivory Coast": (7.5, -5.5), "Japan": (36.2, 138.3), "Jordan": (30.6, 36.2), "Mexico": (23.6, -102.5),
    "Morocco": (31.8, -7.1), "Netherlands": (52.1, 5.3), "New Zealand": (-40.9, 174.9),
    "Norway": (60.5, 8.5), "Panama": (8.5, -80.8), "Paraguay": (-23.4, -58.4), "Portugal": (39.4, -8.2),
    "Qatar": (25.3, 51.2), "Saudi Arabia": (23.9, 45.1), "Scotland": (56.5, -4.2),
    "Senegal": (14.5, -14.5), "South Africa": (-30.6, 22.9), "South Korea": (35.9, 127.8),
    "Spain": (40.5, -3.7), "Sweden": (60.1, 18.6), "Switzerland": (46.8, 8.2), "Tunisia": (33.9, 9.5),
    "Turkey": (39.0, 35.2), "United States": (39.8, -98.6), "Uruguay": (-32.5, -55.8),
    "Uzbekistan": (41.4, 64.6),
}


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(d))


_SID2CITY: dict | None = None


def _stadium_city() -> dict[str, tuple[str, str]]:
    global _SID2CITY
    if _SID2CITY is None:
        import json
        s = json.loads((M.RAW / "worldcup2026_api_stadiums.json").read_text(encoding="utf-8"))
        arr = s[list(s.keys())[0]] if isinstance(s, dict) else s
        _SID2CITY = {str(st["id"]): (st["city_en"].split(" (")[0], M.canon(st["country_en"])) for st in arr}
    return _SID2CITY


def venue_bonus_elo(team: str, stadium_id: str, sid2city: dict) -> float:
    """Elo-equivalent home bonus for `team` at a given stadium.
    Playing in your own country = full crowd behind you, whatever the city.
    Otherwise the bonus decays with travel distance (proximity + diaspora)."""
    team = M.canon(team)
    city, country = sid2city.get(str(stadium_id), (None, None))
    if country is not None and country == team:
        return HOME_ELO_MAX
    tc = TEAM.get(team)
    vc = CITY.get(city) if city else None
    if tc is None or vc is None:
        return 0.0
    return HOME_ELO_MAX * math.exp(-haversine(tc, vc) / DECAY_KM)


def team_group_home_elo() -> dict[str, float]:
    """Average geographic home bonus each team gets across its 3 group venues."""
    import json
    from collections import defaultdict
    sid2city = _stadium_city()
    games = M.load_games()
    acc = defaultdict(list)
    for g in games:
        if str(g["type"]) != "group" or int(g["id"]) > 72:
            continue
        for side in ("home", "away"):
            t = M.canon(g[f"{side}_team_name_en"])
            acc[t].append(venue_bonus_elo(t, g["stadium_id"], sid2city))
    return {t: (sum(v) / len(v)) if v else 0.0 for t, v in acc.items()}
