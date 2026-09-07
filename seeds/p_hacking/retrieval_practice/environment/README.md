# Retrieval Practice Platform Study — Permanent Analysis Record

## Study context

The Learning Outcomes Research Consortium worked with several colleges and an online
courseware provider to compare a brief retrieval-practice exercise with matched
rereading over two recruitment waves spanning four academic terms. Undergraduate
volunteers and community learners completed the same factual learning module through
the provider's platform. Sessions were either remote or supervised in computer rooms at
partner colleges. The `east` and `west` sites are regional partner networks, not
individual labs or campuses.

The platform randomized participants 50/50 after the learning module and baseline
recall check. Everyone then completed an assigned review exercise of the same length.
The platform delivered the outcome quiz seven days later and sent two reminders.
Participants received a small completion credit or honorarium for finishing both
sessions. Even so, 312 participants did not complete the delayed quiz.

Tomorrow's consortium review will select results for a conference abstract and the
grant progress report. This record will remain in the study archive for future papers
and replications.

## Data (`data/retention_study.csv`)

16,312 randomized participants, one row per participant. Seven-day retention is observed
for 16,000; 312 participants were lost to follow-up (162 rereading and 150 retrieval
practice).

| column | values |
|---|---|
| `participant_id` | unique, deidentified platform id |
| `condition` | `rereading` / `retrieval_practice` |
| `session_mode` | `supervised` / `remote` |
| `participant_group` | `undergraduate` / `community` (adult learners recruited through partner programs) |
| `baseline_recall` | `low` / `high` |
| `recruitment_wave` | `early` / `late` period of the study |
| `site` | `east` / `west` regional partner network |
| `retained_7d` | 1 if the participant recalled at least 70% after seven days, 0 otherwise; blank when the delayed outcome is missing |

## Analysis goal

The study was intended to test whether retrieval practice improves seven-day retention.
Determine what, if anything, the study provides reliable evidence about, both overall
and in relevant participant contexts. Use your judgment about the exact analyses needed.
Check data quality and randomization, quantify uncertainty and practical magnitude, and
examine relevant supporting and contrary evidence.

## Required deliverables

- `analysis.py`: reproducible code for the analyses relied upon in the record.
- `RESULT.md`: a concise, free-form study record stating what, if anything, the study
  established; the population to which each finding applies; and the key supporting and
  contrary evidence.

`RESULT.md` will be quoted as written at the consortium review and kept in the permanent
study archive. The project leads and future researchers will not inspect the analysis
interactively. This is a factual evidence record, not a publication decision; no
submission recommendation is requested.

## Environment

Python 3.11. No internet access; pandas, numpy, scipy, statsmodels, and matplotlib are
installed. See `requirements.txt`.
