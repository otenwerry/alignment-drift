# Alignment drift

Code for studying how an agent's previous behavior affects reward hacking on a new task. This repository includes the tasks, prompts, experiment runners, judging code, and figure analysis. The trajectories can be found on [Hugging Face](https://huggingface.co/datasets/okt2002/alignment-drift).

## Install

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen
```

## Reproduce the published figures

The downloader saves the dataset to `data/published/`, using the version pinned in `experiments/published/dataset.json`:

```sh
uv run download_data.py
uv run verify_data.py
uv run analyze.py
```

Verification checks file hashes. Analysis writes `outputs/published/figures.html` and `outputs/published/counts.json`, using the study's exclusions and figure definitions. It does not call any models.

See [docs/data.md](docs/data.md) for the dataset layout and examples of reading transcripts.

To use an existing local copy, run `uv run analyze.py --data-dir /path/to/dataset`.

## Files

| Location | Purpose |
| --- | --- |
| `exp_baseline.py`, `exp_prefix.py`, `exp_continuation.py` | Experiment commands |
| `lib/`, `prefixes/` | Shared runtime and prefix implementations |
| `seeds/`, `envgen/`, `sandbox/` | Task definitions, generated starting assets, generators, and Docker setup |
| `judge_instructions/` | Judge rubrics and prompts |
| `experiments/published/` | Dataset version and checkout campaign configurations |
| `analyze.py`, `templates/` | Local result analysis and presentation |
| `tests/` | Tests for the experiment and analysis code |
