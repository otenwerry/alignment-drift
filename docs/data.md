# Published data

The dataset is hosted at [okt2002/alignment-drift](https://huggingface.co/datasets/okt2002/alignment-drift). `download_data.py` fetches the revision pinned in `experiments/published/dataset.json`, with this layout:

```text
README.md
trajectories.csv
preview.csv
SCHEMA.md
figure_counts.json
manifest.json
logs/<run-id>/...
continuation_prefixes/*.json
activity_logs/...
wikipedia_articles/...
trajectory_ids.json
prefix_selections.json
```

Start with `trajectories.csv`. It contains 4,076 trajectories, of which 3,941 distinct trajectories contribute to the figures. Other rows identify prefix-source candidates and prefix generation. `included_in_figures` defines the plotted cohort; `figure_cells` names each applicable panel/model/condition. `log_path` is relative to the dataset root. The dataset README defines every field and provides a Python loading example.

Inspect `.eval` files preserve transcripts, events, judgments, and metadata. Keep their run-relative structure and associated integrity, cost, and agent-created artifacts intact. Native resume bundles and source snapshots are retained. The original `benchmark_only` flag means an outcome was usable for analysis but its trajectory could not be reused as a prefix; it is not a publication-inclusion flag.

`continuation_prefixes/`, `activity_logs/`, and `wikipedia_articles/` preserve supporting inputs. `trajectory_ids.json` retains stable historical identities, and `prefix_selections.json` retains selection provenance. Unused prefix candidates are intentionally included. Do not infer released membership from the ID registry.

`manifest.json` lists released files, byte sizes, and hashes. Run:

```sh
uv run verify_data.py data/published
uv run analyze.py
```

The manifest omits itself and download-generated state such as `download.json`. The downloader records the pinned HF revision and refuses to mix revisions. Prefix-only downloads contain supporting inputs, not the full corpus required for figure reproduction or complete hash verification.

New experiments write to `data/runs/`. Published analysis reads `data/published/` without changing its registry and writes HTML/caches under `outputs/`.

For maintainers, `uv run export_index.py --data-dir PATH` rebuilds the CSV and figure-count JSON from canonical judgments and curation rules. It verifies the CSV's outcomes and memberships against every figure cell. Regenerate the file manifest after changing released files.

`preview.csv` is the HF table: readable conditions, final outcomes, complete judge-written summaries, and links to logs for the plotted cohort. Rebuild it with `uv run export_preview.py --data-dir PATH --repo-id OWNER/DATASET --revision COMMIT`, using the revision containing the logs. The technical index remains `trajectories.csv`.
