# Published experiment definitions

`dataset.json` pins [okt2002/alignment-drift](https://huggingface.co/datasets/okt2002/alignment-drift) to the full commit hash used by the downloader.

`checkout_continuations.json` captures all 46 exact prefix definitions from the completed checkout campaign, including model, harness, source task, treatment, filename, and SHA-256. It is configuration, not results. `checkout_baselines.json` records the matching baseline settings and seed hashes. The supported public runners are `exp_continuation.py` and `exp_baseline.py`.

The complete figure-to-condition mapping remains in `lib/post_figure_data.py`; archival rules remain in `viewer_old_runs.json` and `lib/old_runs.py`. These are reused by `analyze.py`. The published dataset preserves referenced inputs and all plotted judgment sources. Some intentionally omitted or failed trajectories can leave plotted denominators below their requested sample size.
