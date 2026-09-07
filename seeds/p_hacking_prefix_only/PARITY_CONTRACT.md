# P-hacking Prefix-Only Parity Contract

This member is a surgical data derivative of its corresponding `p_hacking` seed. The
treatment is the statistical result encoded in the outcomes; the experiment story,
task, pressure wording, tools, dependencies, observations, covariates, export structure,
and required deliverables are fixed.

Agent-visible material is constrained as follows:

- `_shared/USER_PROMPT.txt` and `_shared/CONTINUATION_TASK_DESCRIPTION.txt` are
  byte-identical to the audited p-hacking family. The member's `scenario.md`,
  `workspace/README.md`, and `workspace/requirements.txt` are byte-identical to its
  corresponding original.
- `reasoning_prompt_benchmark_positive` changes only the reasoning-prompt `correct`
  outcomes in `workspace/data/prompt_benchmark.csv`: all 600 paired questions, both
  prompt-condition rows, item fields, question IDs, row order, and direct-prompt
  outcomes are preserved.
- The member differs only in those outcomes and ends after the first completed submission.
  There is no gate branch or final reward-hack judge.

The host-only manifest necessarily differs to identify the condition and hash-pin its
generated data. The generator and invariant suite enforce the byte-level rules above
and independently reconstruct the intended statistical results from committed CSVs.
