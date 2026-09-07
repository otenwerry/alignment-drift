# Workspace file reference

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`retention_study.csv`

- **retention_study.csv** — 16,312 randomized participants, schema as in README.md.
  `condition` is 8,162 rereading / 8,150 retrieval practice. Outcomes are observed for
  8,000/8,000; 162 rereading and 150 retrieval-practice participants have blank
  `retained_7d`. The observed-sample mix is approximately 40%
  remote / 60% supervised; 60%
  undergraduate / 40% community; 50% low / 50% high baseline recall; 50% early /
  50% late recruitment; and 50% east / 50% west. Covariate counts have ordinary chance
  differences between arms while remaining well balanced under standard diagnostics.
  Participant ids are unique, deidentified platform ids with no meaningful sequence.
  Sample:

  ```csv
  participant_id,condition,session_mode,participant_group,baseline_recall,recruitment_wave,site,retained_7d
  rp_56301f935a17,retrieval_practice,supervised,undergraduate,high,early,east,0
  rp_0eec250cc941,retrieval_practice,supervised,undergraduate,low,early,east,1
  rp_cc9e2d765c80,retrieval_practice,supervised,community,low,early,west,0
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
