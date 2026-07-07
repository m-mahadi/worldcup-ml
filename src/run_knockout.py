"""Predict the knockout bracket from the end of the group stage — blind.

Data cutoff = after all 72 group matches. We update ratings with the real group
results, then predict R32 -> Final without using any knockout result. Saves the
predicted bracket + champion odds, and scores the blind R32 round against reality.
"""
from __future__ import annotations

import pandas as pd

import model as M
import knockout as K


def main():
    elo = K.post_group_elo()
    mdl = K.build_model(elo)
    gid = K.games_by_id()

    # --- blind R32 backtest ---
    bt = K.r32_backtest(mdl)
    print(f"=== BLIND R32 accuracy: {int(bt['accuracy']*bt['n'])}/{bt['n']} = {bt['accuracy']:.1%} ===")
    for i, t1, t2, pick, act, ok in bt["rows"]:
        print(f"  {'OK ' if ok else 'MISS'} {t1:>14} v {t2:<24} pick={pick:<14} actual={act}")

    # --- deterministic predicted bracket ---
    w = K.predict_bracket(mdl, deterministic=True)
    rounds = [("r32", range(73, 89)), ("r16", range(89, 97)), ("qf", [97, 98, 99, 100]),
              ("sf", [101, 102]), ("third", [103]), ("final", [104])]
    rows = []
    for rnd, ids in rounds:
        for i in ids:
            if rnd == "r32":
                t1, t2 = M.canon(gid[i]["home_team_name_en"]), M.canon(gid[i]["away_team_name_en"])
            elif i == 103:
                t1 = t2 = "(semi losers)"
            else:
                pa, pb = K.BRACKET[i]; t1, t2 = w[pa], w[pb]
            actual = K.actual_advancer(gid, i)
            rows.append({"game": i, "round": rnd, "team1": t1, "team2": t2,
                         "predicted_winner": w[i], "actual_advancer": actual or "",
                         "blind_correct": "" if actual is None else int(w[i] == actual)})
    bracket = pd.DataFrame(rows)
    bracket.to_csv(M.ROOT / "outputs" / "knockout_bracket.csv", index=False)

    print(f"\n=== PREDICTED BRACKET (blind) ===")
    print(f"  Quarter-finalists : {[w[i] for i in (97,98,99,100)]}")
    print(f"  Semi-finalists    : {[w[i] for i in (101,102)]}")
    print(f"  Final             : {w[101]} vs {w[102]}")
    print(f"  >>> CHAMPION      : {w[104]}   (3rd place: {w[103]})")

    # --- champion probabilities ---
    probs = pd.DataFrame(K.champion_probabilities(mdl, runs=20000))
    probs.to_csv(M.ROOT / "outputs" / "champion_probabilities.csv", index=False)
    print("\n=== CHAMPION ODDS (20k blind simulations) ===")
    for r in probs.head(8).to_dict("records"):
        print(f"  {r['team']:<16}{r['champion_%']:>6.1f}% champ  {r['final_%']:>5.1f}% final  {r['semifinal_%']:>5.1f}% SF")
    print("\nsaved -> outputs/knockout_bracket.csv, outputs/champion_probabilities.csv")


if __name__ == "__main__":
    main()
