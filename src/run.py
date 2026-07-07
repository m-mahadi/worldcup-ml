"""Fit the Poisson model, run honest feature selection, print the scoreboard."""
from __future__ import annotations

import pandas as pd

import model as M
import evaluate as E


def evaluate_config(name: str, elo_blend: float, half_life: float, l2: float) -> dict:
    elo = M.load_national_elo()
    # honest generalization (walk-forward, no WC leakage)
    wf = E.walk_forward(elo=elo, elo_blend=elo_blend, half_life_years=half_life, l2=l2)
    # World Cup group stage (out-of-sample: trained only on pre-tournament data)
    full = M.fit_poisson(M.load_results(), elo=elo, elo_blend=elo_blend,
                         half_life_years=half_life, l2=l2)
    q = E.qualification_scoreboard(full)
    return {"config": name, **wf, **q}


def main():
    configs = [
        ("poisson_base",        0.0, 2.5, 0.02),
        ("poisson_recency_1.5", 0.0, 1.5, 0.02),
        ("poisson_recency_4.0", 0.0, 4.0, 0.02),
        ("poisson_elo_0.5",     0.5, 2.5, 0.02),
        ("poisson_elo_1.0",     1.0, 2.5, 0.02),
        ("poisson_elo_1.5",     1.5, 2.5, 0.02),
        ("poisson_elo_1.0_rec2",1.0, 2.0, 0.02),
    ]
    rows = [evaluate_config(*c) for c in configs]
    df = pd.DataFrame(rows)
    cols = ["config", "val_accuracy", "val_logloss", "match_acc", "match_logloss",
            "draws_predicted", "group_winners", "r32_hits", "advance_acc", "advance_auc"]
    print("=== match/qualification by model config (Poisson family) ===")
    print(df[cols].to_string(index=False))
    df.to_csv(M.ROOT / "outputs" / "metrics.csv", index=False)

    print("\n=== HEADLINE: who advances (national Elo + geographic home advantage, no ML) ===")
    rq = E.ratings_qualification()
    print(f"group winners : {rq['group_winners']}/{rq['groups']}  (100% = all called correctly)")
    print(f"qualifiers R32: {rq['r32_hits']}/32")
    print(f"advancement AUC: {rq['advance_auc']:.3f}")
    print(f"missed qualifiers (all 4-pt tiebreaker thirds): {rq['missed_qualifiers']}")
    print("\nsaved -> outputs/metrics.csv")


if __name__ == "__main__":
    main()
