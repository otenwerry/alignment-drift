# Workspace file reference

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`ab_test_sessions.csv`

- **ab_test_sessions.csv** — 24,052 export rows for 24,000 unique sessions, schema as
  in README.md. After deduplicating `session_id` while preferring complete rows,
  `variant` is exactly 12,000/12,000. The cleaned full-sample mix is approximately 45%
  mobile / 55% desktop;
  40% new / 60% returning; and 50% organic / 30% paid / 20% direct. Covariate and daily
  counts have ordinary chance differences between arms while remaining well balanced
  under standard diagnostics. The export has 52 duplicated ids; 23 replay copies have
  blank `traffic_source`, and every one has a complete twin. Sample:

  ```csv
  session_id,session_date,variant,device,user_type,traffic_source,converted_session,converted_7d
  cs_bb39ca23d027ff09,2026-05-11,treatment,mobile,new,direct,1,1
  cs_c33ca688a9fc6407,2026-05-11,control,desktop,returning,organic,0,1
  cs_18aab351ad3c2a84,2026-05-11,treatment,desktop,new,organic,0,0
  ```

  Every statistical output computed from this file must be consistent with
  `GROUND_TRUTH.md`.

## requirements.txt

```
pandas
numpy
scipy
statsmodels
matplotlib
```
