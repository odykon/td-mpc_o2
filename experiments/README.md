# experiments/

All experiments from 2026-11 onward. Older results live in `runs/`, `logs/` and `models/`; those folders are frozen.

```
experiments/<algo>/<task>/<YYYY-MM-DD_slug>/<variant>_s<seed>/
```

- `algo`: `tdmpc`, `o2`, `o2_base`, `dcem_toy`, ...
- `task`: e.g. `cheetah-run`
- `YYYY-MM-DD_slug`: the date the experiment started, plus a short name
- `variant`: only the settings that differ within the experiment, e.g. `ar3` or `ar3_std-graph`. If there is no variant, the folder is just `s<seed>`.

Naming: lowercase, `-` inside a token, `_` between tokens, no spaces.

Every run folder contains:
- `config.yaml`: the full resolved cfg
- `meta.json`: `commit`, `dirty` (tracked files only), `date`, `host`, `cmd`, and optional `notes`

Everything else is free-form: `*.csv` for metrics, `*.pt`/`*.pth` for weights and buffers.
Files that summarize a whole experiment (summary CSVs, `plots/`) go in the experiment folder itself.

Create run folders with `experiments_utils.make_run_dir(...)`. It refuses to overwrite an existing run.
