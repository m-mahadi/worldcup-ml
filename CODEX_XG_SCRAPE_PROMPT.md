# Task for Codex: scrape per-match xG + performance stats for the 2026 World Cup

## What I need
Per-match advanced stats for the completed 2026 FIFA World Cup matches, primarily
**expected goals (xG)** for both teams, plus supporting quality stats if the source
shows them. This feeds a prediction model that uses performance quality, not just
scorelines.

NOTE: this is a neutral-venue tournament — there is NO home/away team. The two
teams in each match are just `team1` and `team2` (the order they are listed in the
fixture, nothing more). Keep that neutral labelling in the output.

## Input
The exact list of matches to find is here (94 rows, already keyed to my internal IDs):

`D:\worldcup-ml\data\raw\fixtures_for_xg.csv`

Columns: `match_id, stage, date, team1, team2, team1_score, team2_score`
(dates are MM/DD/YYYY; this is the 2026 World Cup hosted by USA/Canada/Mexico.)

## Output (write exactly this file)
`D:\worldcup-ml\data\raw\match_xg_stats.csv`

One row per match you find, with these columns (leave a cell blank if a stat isn't
available, but **xG is the priority** — get that even if nothing else):

```
match_id,team1,team2,date,src_team1_score,src_team2_score,
team1_xg,team2_xg,team1_shots,team2_shots,team1_sot,team2_sot,
team1_big_chances,team2_big_chances,team1_poss,team2_poss,source,match_url,note
```

- `match_id` — copy from the input CSV (this is the join key, do not invent new ids).
- `team1`/`team2` — same two teams as the input row, in the SAME order (team1 = the
  input's team1). These are just labels for the two sides, not home/away.
- `src_team1_score`/`src_team2_score` — the score AS SHOWN on the stats site, for the
  team you mapped to team1 / team2. This lets me verify the site's match is the same
  match. **If the site's score does not match the input CSV's score, still record it
  but put `SCORE_MISMATCH` in `note`.**
- `team1_xg`/`team2_xg` — expected goals, decimals (e.g. 1.73). Required.
- `team1_sot`/`team2_sot` — shots on target.
- `team1_big_chances`/`team2_big_chances` — "big chances" if the site has them.
- `team1_poss`/`team2_poss` — possession % as integers.
- `source` — which site the row came from (e.g. `sofascore`, `fbref`, `fotmob`).
- `match_url` — the page you pulled it from, for auditing.

## Sources (try in this order)
1. **Sofascore** — sofascore.com. Each match page has an xG stat. There is a JSON
   endpoint pattern `https://api.sofascore.com/api/v1/event/{eventId}/statistics`
   and a search endpoint to find the event id by teams+date. This is usually the
   most scriptable. Look for the "Expected goals (xG)" row.
2. **FBref** — fbref.com. The World Cup "Scores & Fixtures" table has an `xG` column,
   and each match report has full shot/xG data. It 403s plain scripts, so use a real
   browser session / proper headers if you have browser access.
3. **FotMob** — fotmob.com match pages show xG under stats.

Cross-check where cheap; if two sources disagree on xG, prefer Sofascore and note it.

## Rules
- Match by **team names + date** (team-name spellings will vary — Türkiye/Turkey,
  DR Congo/Congo DR, USA/United States, etc. — match on the obvious equivalent).
  Map each site's two teams to team1/team2 in the SAME order as the input CSV.
- Only the 94 matches in the input CSV. Group stage matters most; knockouts are a bonus.
- Do NOT fabricate numbers. Blank is better than a guess. If a match genuinely has no
  xG anywhere, leave the xG cells blank and put `NO_XG_FOUND` in `note`.
- Keep the output strictly to the schema above so it loads without cleanup.
- When done, print: how many of the 94 matches got xG, how many had SCORE_MISMATCH,
  and which sources were used.

## Why the score check matters
My fixture data comes from one feed and I can't independently verify every scoreline.
The `src_team1_score`/`src_team2_score` + `SCORE_MISMATCH` flag let me detect if the
stats site is describing a different match, so I don't join bad data. Please be
strict about this.
