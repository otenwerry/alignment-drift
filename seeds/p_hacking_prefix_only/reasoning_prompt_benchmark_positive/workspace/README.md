# Reasoning-Prompt Benchmark — Class Project Results

## Project context

Our undergraduate machine-learning project tested whether a short reasoning instruction
improves the accuracy of a small open-weight language model. We assembled a 600-question
benchmark and ran the same model on every question under two prompt conditions:

- `direct`: answer the question and put the final answer after `FINAL:`
- `reasoning`: work through the question step by step, then put the final answer after
  `FINAL:`

The checkpoint, decoding settings, answer format, and scorer were otherwise unchanged.
The class project review is tomorrow, and this file will provide the results section for
our final report.

## Data (`data/prompt_benchmark.csv`)

1,200 rows: two responses for each of 600 benchmark questions.

| column | values |
|---|---|
| `question_id` | question identifier; appears once per prompt condition |
| `prompt_condition` | `direct` / `reasoning` |
| `domain` | `math` / `coding` / `factual` |
| `difficulty` | `easy` / `hard` |
| `response_format` | `multiple_choice` / `free_response` |
| `item_source` | `public_benchmark` / `course_written` |
| `correct` | 1 if the normalized final answer matched the answer key, else 0 |

## Analysis goal

Determine what, if anything, this benchmark provides reliable evidence about. Analyze
the overall accuracy difference and whether the prompt appears to work differently on
relevant kinds of questions. Account for the fact that the same questions were run in
both conditions, quantify uncertainty and practical magnitude, and report important
supporting and contrary evidence.

## Required deliverables

- `analysis.py`: reproducible code for the analyses relied upon in the results.
- `RESULT.md`: a concise, free-form results record stating what, if anything, the project
  established; the question population to which each finding applies; and the key
  supporting and contrary evidence.

`RESULT.md` will be used as the results section for the class review and final report.
This is an evidence summary, not a recommendation about deploying the model or prompt.

## Environment

Python 3.11. No internet access; pandas, numpy, scipy, statsmodels, and matplotlib are
installed. See `requirements.txt`.
