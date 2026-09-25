# OpenFNIRS-sensorimotor

A large-scale fNIRS sensorimotor benchmark for fNIRS classification methods.
**N = 100 participants** performing right-hand **hand clenching** vs **finger
tapping**, recorded across **13 channels** over the left motor cortex
(HbO + HbR), sampled at 10.173 Hz.

**Live dashboard:** https://s-klinkhammer.github.io/OpenFNIRS-sensorimotor/

## Repository contents

| Path | Purpose |
|---|---|
| `docs/` | **Static dashboard** for browsing the preprocessed data — served by GitHub Pages |
| `docs/data/` | Precomputed bundle the dashboard reads (grand average, per-subject means, single-trial tensors) |
| `build_data.py` | Builds the dashboard data bundle (`public/data/`) directly from the BIDS dataset (raw `events.tsv` + preprocessed `.snirf` derivatives) |
| `flatten_raw_for_satori.py` | Optional helper: copies every subject's raw `.snirf` file into one flat folder, so Satori's Load-node file dialog can select across subjects instead of navigating into each `sub-XXX/nirs/` folder — see `preprocessing/preprocessing_readme.md` / `quality/quality_readme.md` |
| `BIDS_scripts/` | Scripts for converting raw recordings into BIDS format |
| `preprocessing/` | Satori workflow used to turn BIDS raw data into preprocessed derivatives — see `preprocessing/preprocessing_readme.md` |
| `classification/` | Feature extraction and Riemannian-ensemble classification pipeline used in the manuscript (Section 4.2) — see `classification/classification_readme.md` |
| `environment.yml` | Conda environment for the analysis scripts |

## Getting the data

The full dataset — raw recordings and preprocessed derivatives, as one BIDS
tree — is published on Zenodo (DOI: **TODO — add your Zenodo DOI/link
here**).

1. Download the archive and unzip it into `data/` at the repository root.
2. Depending on how the zip is packaged you'll end up with either
   `data/bids_fnirs_sensorimotor_dataset/` or the BIDS folders directly
   under `data/` — **both work**, every script in this repo checks for the
   first and falls back to the second automatically. Either way you should
   end up with both of these present:
   ```
   data/[bids_fnirs_sensorimotor_dataset/]sub-XXX/nirs/*_events.tsv                       <- raw
   data/[bids_fnirs_sensorimotor_dataset/]derivatives/nirs-preproc/sub-XXX/nirs/*.snirf    <- preprocessed
   ```

That's it — no manual path configuration needed for any Python script in
this repo (`build_data.py`, `classification/generate_npy.py`,
`glm_analysis/generate_msd.py`, `glm_analysis/move_multistudy_files.py`).
Only the Satori `.flow` workflows in `preprocessing/` and `quality/` need a
path set manually, since Satori itself doesn't auto-detect folders — see
those READMEs for which node to point where.

## Live dashboard

The static dashboard under `docs/` is fully self-contained — HTML, CSS, JS,
and binary data fetched from the same origin. It runs without any build step
and embeds via `<iframe>` if you'd like to drop it into another page.

### Run locally

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

## Reproducing the analysis

1. Get the data as described in "Getting the data" above (you only need the
   `derivatives/nirs-preproc/` part of it for this pipeline).
2. Set up the environment and run the pipeline end-to-end:

   ```bash
   conda env create -f environment.yml
   conda activate respra
   python classification/generate_npy.py
   python classification/publish_riemann_ensemble.py
   ```

Outputs land in `results/publish_ensemble/`.

## Reproducing the dashboard bundle

`build_data.py` (repo root) regenerates the data the live dashboard reads.
It needs both the raw `events.tsv` files and the preprocessed `.snirf`
derivatives — i.e. the full drop-in described in "Getting the data" above,
nothing extra. Then:

```bash
python build_data.py
```

This writes `meta.json`, `grand_avg.json`, `subject_avg.bin`, and one
`.bin`/`_labels.json` pair per subject into `public/data/`.

> **Note:** this script writes to `public/data/`, while the "Live dashboard"
> section above refers to `docs/` as what GitHub Pages serves. If your
> GitHub Pages source is `docs/` and it isn't kept in sync with `public/`
> some other way (symlink, copy step, build tool), double-check that —
> otherwise re-running this script won't update the live dashboard.

## Acknowledgments

Parts of the code in this repository (documentation, repository structure,
and debugging of the classification and preprocessing scripts) were
developed with the assistance of Claude (Anthropic).

## Citation

```bibtex
@dataset{openfnirs_sensorimotor_2026,
  author  = {Klinkhammer, S. and N{\"a}her, T. and Raible, S.
             and L{\"u}hrs, M. and Klein, F. and Sorger, B.},
  title   = {OpenFNIRS-sensorimotor: a large-scale fNIRS dataset for
             benchmarking advanced classification methods},
  year    = {2026},
  version = {1.0},
}
```

## License

Code: MIT. Data: refer to the linked raw-data release for licensing.
