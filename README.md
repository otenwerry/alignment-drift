# Alignment drift

Code for studying how an agent's previous behavior affects reward hacking on a new task. This repository includes the tasks, prompts, experiment runners, judging code, and figure analysis. The [trajectories are on Hugging Face](https://huggingface.co/datasets/okt2002/alignment-drift).

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

The dataset's `trajectories.csv` lists each record's model, condition, outcome, figure membership, and log path. See [docs/data.md](docs/data.md) for the file layout and examples of reading transcripts.

To use an existing local copy, run `uv run analyze.py --data-dir /path/to/dataset`. Downloaded data and generated output are ignored by Git.

## Run a small experiment

Local experiments require Docker and credentials for the agent and judge providers, supplied through your shell or a local `.env`. Model mappings are in `lib/model_catalog.py`.

The examples below use `--dry-run` to show the plan. Remove it to run the experiment.

Generate one baseline trajectory:

```sh
uv run exp_baseline.py --targets=deepseek-v4-pro \
  --seed-dir=p_hacking --seeds=checkout_redesign --epochs=1 \
  --harness=production --pressure=low --compute=local --dry-run
```

Build an ML prefix without access to test labels:

```sh
uv run exp_prefix.py ml --targets=deepseek-v4-pro \
  --seeds=demand_forecasting --epochs=1 \
  --harness=production --compute=local --dry-run
```

Continue a downloaded or newly generated prefix on another task:

```sh
uv run exp_continuation.py --treatment=my-condition \
  --prefix-files=data/published/continuation_prefixes/PREFIX.json \
  --seed-dir=p_hacking --seeds=checkout_redesign --epochs=1 \
  --harness=production --pressure=low --compute=local --dry-run
```

Replace `PREFIX.json` with a downloaded or generated prefix. The prefix records the agent and reasoning settings; use its recorded harness and a different destination task.

`production` uses native agent harnesses with API billing; `subscription` uses subscription authentication; `simple` uses the Inspect tool loop. These settings can change experimental behavior. The commands also support AWS execution. Use `--help` for options and `uv run exp_prefix.py --help` for other prefix types.

`download_data.py --prefixes-only` downloads just the continuation inputs. Reproducing figures needs the full dataset.

New logs are saved to `data/runs/logs/` and prefixes to `data/runs/continuation_prefixes/`. Set `ENVIRONMENTS_DATA_ROOT` to change the data directory or `ENVIRONMENTS_OUTPUT_ROOT` to change the output directory.

## Files

| Location | Purpose |
| --- | --- |
| `exp_baseline.py`, `exp_prefix.py`, `exp_continuation.py` | Experiment commands |
| `lib/`, `prefixes/` | Shared runtime and prefix implementations |
| `seeds/`, `envgen/`, `sandbox/` | Task definitions, generated starting assets, generators, and Docker setup |
| `judge_instructions/` | Judge rubrics and prompts |
| `experiments/published/` | Dataset version and checkout campaign configurations |
| `analyze.py`, `templates/` | Local result analysis and presentation |
| `export_index.py`, `export_preview.py` | Dataset index and HF preview generation |
| `tests/` | Tests for the experiment and analysis code |

## Tests

```sh
uv run pytest --ignore=tests/test_real_sandbox_smoke.py
```
