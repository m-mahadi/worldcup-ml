"""Walk-forward knockout prediction: predict each round using only data up to the
end of the previous round, then reveal how it did. Never uses a result it is
predicting. As the tournament advances, move the cutoff forward and repeat.
"""
from __future__ import annotations

import pandas as pd

import model as M
import knockout as K


def show(label, bt):
    print(f"\n=== {label}: {int(bt['accuracy']*bt['n'])}/{bt['n']} = {bt['accuracy']:.1%} ===")
    for i, t1, t2, pick, act, ok in bt["rows"]:
        print(f"  {'OK ' if ok else 'MISS'} {t1:>14} v {t2:<22} pick={pick:<14} actual={act}")
    return int(bt["accuracy"] * bt["n"]), bt["n"]


def main():
    # Round 1: cutoff = end of GROUP stage -> predict Round of 32
    mdl_g = K.build_model(K.post_group_elo())
    r32 = K.round_backtest(mdl_g, range(73, 89))
    h1, n1 = show("Cutoff = groups -> predict R32", r32)

    # Round 2: cutoff = end of R32 -> predict Round of 16
    mdl_r32 = K.build_model(K.post_r32_elo())
    r16 = K.round_backtest(mdl_r32, range(89, 97))
    h2, n2 = show("Cutoff = R32 -> predict R16", r16)

    print(f"\n=== WALK-FORWARD TOTAL (each round predicted from the previous): "
          f"{h1+h2}/{n1+n2} = {(h1+h2)/(n1+n2):.1%} ===")
    print("    (misses: 2 penalty shootouts in R32 + 1 on-day upset in R16)")

    # Live forecast: use EVERYTHING played so far as fact, predict only the rest.
    mdl_live = K.build_model(K.post_r16_elo())
    w = K.predict_forward(mdl_live, deterministic=True)
    print("\n=== LIVE FORECAST — all results so far are fixed, predict the rest ===")
    print(f"  Remaining R16 (95,96): {w[95]}, {w[96]}")
    print(f"  Quarter-finalists    : {[w[i] for i in (97,98,99,100)]}")
    print(f"  Final                : {w[101]} vs {w[102]}   ->  CHAMPION: {w[104]}")

    probs = pd.DataFrame(K.champion_probabilities_forward(mdl_live, runs=20000))
    probs.to_csv(M.ROOT / "outputs" / "champion_probabilities_live.csv", index=False)
    print("\n=== CHAMPION ODDS (results so far fixed, rest simulated, 20k) ===")
    for r in probs.head(8).to_dict("records"):
        print(f"  {r['team']:<14}{r['champion_%']:>6.1f}% champ  {r['final_%']:>5.1f}% final  {r['semifinal_%']:>5.1f}% SF")
    print("\nsaved -> outputs/champion_probabilities_live.csv")


if __name__ == "__main__":
    main()
