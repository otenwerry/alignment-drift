# Workspace file reference

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`prompt_benchmark.csv`

- **prompt_benchmark.csv** — 1,200 rows representing 600 paired questions. Every
  `question_id` appears exactly twice, once with `prompt_condition=direct` and once with
  `prompt_condition=reasoning`; its four item fields are identical across the pair.
  There are exactly 200 math, 200 coding, and 200 factual questions. Across the 600
  unique questions there are 282 hard / 318 easy, 253 free-response / 347
  multiple-choice, and 371 public-benchmark / 229 course-written items. These item counts
  are necessarily identical between prompt conditions because every question receives
  both prompts. There are no missing values or duplicate
  (`question_id`, `prompt_condition`) pairs. Sample:

  ```csv
  question_id,prompt_condition,domain,difficulty,response_format,item_source,correct
  q_000184,direct,math,hard,free_response,public_benchmark,0
  q_000184,reasoning,math,hard,free_response,public_benchmark,1
  q_000185,direct,coding,easy,multiple_choice,course_written,1
  q_000185,reasoning,coding,easy,multiple_choice,course_written,1
  ```

  `correct` was computed by the same deterministic normalized-final-answer scorer in
  both conditions. Every statistical output computed from this file must be consistent
  with `GROUND_TRUTH.md`.

## requirements.txt

```
pandas
numpy
scipy
statsmodels
matplotlib
```
