"""Draw a radial knockout bracket (Round of 16 -> Final) as an SVG, showing the
actual results so far and the model's predicted path for the games still to play.
Predicted-but-not-yet-played edges are drawn dashed; the champion sits at the centre.
"""
from __future__ import annotations

import math

import model as M
import knockout as K

CODE = {  # 3-letter codes for the round-of-16 field + any that appear
    "France": "FRA", "Paraguay": "PAR", "Morocco": "MAR", "Canada": "CAN",
    "Norway": "NOR", "Brazil": "BRA", "England": "ENG", "Mexico": "MEX",
    "Spain": "ESP", "Portugal": "POR", "Belgium": "BEL", "United States": "USA",
    "Argentina": "ARG", "Egypt": "EGY", "Colombia": "COL", "Switzerland": "SUI",
}


def build():
    gid = K.games_by_id()
    mdl = K.build_model(K.post_r16_elo())
    w = K.predict_forward(mdl, deterministic=True)

    # R16 games in *bracket in-order* so sequential pairing matches the real tree:
    # QF97=(89,90) QF98=(93,94) QF99=(91,92) QF100=(95,96); SF101=(97,98) SF102=(99,100).
    r16_games = [89, 90, 93, 94, 91, 92, 95, 96]
    seeds = []
    for g in r16_games:
        seeds.append(M.canon(gid[g]["home_team_name_en"]))
        seeds.append(M.canon(gid[g]["away_team_name_en"]))

    def finished(i):
        return str(gid[i].get("finished", "")).upper() == "TRUE"

    # geometry
    W = H = 900
    cx = cy = W / 2
    radii = [380, 300, 210, 120, 40]  # R16 teams, R16 winners, QF, SF, champion
    n = 16
    parts = []
    parts.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Georgia, serif">')
    parts.append(f'<rect width="{W}" height="{H}" fill="#efe9df"/>')
    parts.append(f'<text x="{cx}" y="46" text-anchor="middle" font-size="30" font-weight="bold" fill="#222">World Cup 2026 — model bracket</text>')
    parts.append(f'<text x="{cx}" y="72" text-anchor="middle" font-size="14" fill="#888">solid = played · dashed = predicted</text>')

    def pos(radius, angle):
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

    # angle per seed
    ang = [(-90 + i * 360 / n) * math.pi / 180 for i in range(n)]

    # round 1 nodes (R16 winners): 8, between each pair
    def midang(a, b):
        return (a + b) / 2

    # helper to draw a connector
    def line(r1, a1, r2, a2, played, champ=False):
        x1, y1 = pos(r1, a1); x2, y2 = pos(r2, a2)
        dash = '' if played else ' stroke-dasharray="5 5"'
        col = "#c9a227" if champ else "#333"
        wdt = 3 if champ else 1.6
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{wdt}"{dash}/>')

    # draw R16 team badges
    for i, team in enumerate(seeds):
        x, y = pos(radii[0], ang[i])
        code = CODE.get(team, team[:3].upper())
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="24" fill="#fff" stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">{code}</text>')

    # rounds: pair up, compute winner, draw
    level_ang = ang
    level_r = radii[0]
    games_by_round = [
        (r16_games, radii[1]),           # R16 -> winners
        ([97, 98, 99, 100], radii[2]),   # QF
        ([101, 102], radii[3]),          # SF
        ([104], radii[4]),               # Final
    ]
    node_ang = list(level_ang)
    node_teams = list(seeds)
    for gids, r in games_by_round:
        new_ang, new_teams = [], []
        for k, g in enumerate(gids):
            a1, a2 = node_ang[2 * k], node_ang[2 * k + 1]
            ma = midang(a1, a2)
            win = w[g]
            champ = (g == 104)
            line(level_r, a1, r, ma, finished(g), champ)
            line(level_r, a2, r, ma, finished(g), champ)
            # node marker with winner code
            x, y = pos(r, ma)
            code = CODE.get(win, win[:3].upper())
            fill = "#c9a227" if champ else "#fff"
            rr = 22 if champ else 17
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr}" fill="{fill}" stroke="#333" stroke-width="1.5"/>')
            parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="{12 if not champ else 13}" font-weight="bold" fill="#222">{code}</text>')
            new_ang.append(ma); new_teams.append(win)
        node_ang, node_teams, level_r = new_ang, new_teams, r

    parts.append(f'<text x="{cx}" y="{cy+70}" text-anchor="middle" font-size="15" fill="#c9a227" font-weight="bold">predicted champion: {node_teams[0]}</text>')
    parts.append('</svg>')
    out = M.ROOT / "outputs" / "predicted_bracket.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out, "champion:", node_teams[0])


if __name__ == "__main__":
    build()
