# OpenFNIRS-sensorimotor

A large-scale fNIRS sensorimotor benchmark for fNIRS classification methods.
**N = 109 participants** performing right-hand **hand clenching** vs **finger
tapping**, recorded across **13 channels** over the left motor cortex
(HbO + HbR), sampled at 10.173 Hz.

**Live dashboard:** https://s-klinkhammer.github.io/OpenFNIRS-sensorimotor/
*(updates after the first GitHub Pages deploy)*

## Repository contents

| Path | Purpose |
|---|---|
| `public/` | **Static dashboard** for browsing the preprocessed data — served by GitHub Pages |
| `public/data/` | Precomputed bundle the dashboard reads (grand average, per-subject means, single-trial tensors) |
| `build_data.py` | Pipeline that produces `public/data/` from raw per-subject `.npy` tensors |
| `publish_riemann_ensemble.py` | Riemannian-ensemble classification pipeline used in the manuscript (Section 4.2) |
| `environment.yml` | Conda environment for the analysis scripts |

The raw fNIRS dataset (BIDS / SNIRF) is published separately — see the
manuscript or the dashboard's *Download data* link for the persistent URL.

## Live dashboard

The static dashboard under `public/` is fully self-contained — HTML, CSS, JS,
and binary data fetched from the same origin. It runs without any build step
and embeds via `<iframe>` if you'd like to drop it into another page.

### Run locally

```bash
cd public
python3 -m http.server 8000
# open http://localhost:8000
```

### Deploy on GitHub Pages

After pushing this repository to GitHub:

> Settings → Pages → Source: *Deploy from a branch* → Branch: `main`,
> Folder: `/public` → Save

Live URL appears at the top of the Pages settings panel within ~1 minute.

## Reproducing the analysis

The Riemannian-ensemble classification reported in the manuscript is run
end-to-end from `publish_riemann_ensemble.py`:

```bash
conda env create -f environment.yml
conda activate respra
python publish_riemann_ensemble.py
```

Outputs land in `results/publish_ensemble/` (gitignored).

## Rebuilding the dashboard data bundle

`public/data/` is checked in, so you only need to rebuild it when the
upstream raw `.npy` tensors change.

```bash
# place per-subject files at ./data/Pxxx/Pxxx_SD_X.npy and Pxxx_SD_y.npy
python3 build_data.py
```

This regenerates `public/data/meta.json`, `public/data/grand_avg.json`,
`public/data/subject_avg.bin`, and one `public/data/subjects/Pxxx.bin`
per participant. Requires only `numpy`.

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

Code: MIT (placeholder — confirm before publishing).
Data: refer to the linked raw-data release for licensing.
