# NP02 Purity Monitor + Test Pulse analysis

This repo contains analysis scripts for the NP02 liquid argon purity monitor at CERN.

It includes:
- Purity monitor lifetime/purity extraction (NP02 PM)
- Test pulse (TP) fitting and plotting (Crystal Ball fits)


## Folder layout

- `tp/`: test pulse analysis (recommended entrypoints)
  - `TP_fit_CB_to_json.py`: fit TP measurements and write a JSON cache (incremental)
  - `TP_plot_CB_from_json.py`: plotting-only from the JSON cache (no refitting)

## Quickstart

Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical workflow (Test Pulse)

1) Fit once (writes JSON cache and optional per-measurement PNGs):

```bash
python tp/TP_fit_CB_to_json.py --force
```

2) Re-plot any time without refitting:

```bash
python tp/TP_plot_CB_from_json.py
```

Both scripts support `--start` / `--end` time window arguments.

## Data

This repo does **not** include large raw datasets or output plots.
Most scripts expect your local data folder layout (see inline constants like `ROOT_DIR` or environment variables where present).

