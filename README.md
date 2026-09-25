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
| `build_data.py` | Pipeline that produces `docs/data/` from raw per-subject `.npy` tensors |
| `BIDS_scripts/` | Scripts for converting raw recordings into BIDS format |
| `preprocessing/` | Satori workflow used to turn BIDS raw data into preprocessed derivatives — see `preprocessing/preprocessing_readme.md` |
| `classification/` | Feature extraction and Riemannian-ensemble classification pipeline used in the manuscript (Section 4.2) — see `classification/classification_readme.md` |
| `environment.yml` | Conda environment for the analysis scripts |

The raw and preprocessed (derivatives) fNIRS dataset (BIDS / SNIRF) is
published separately on Zenodo — see `classification/classification_readme.md`
and `preprocessing/preprocessing_readme.md` for exactly where to place it.

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

1. Download the preprocessed derivatives from Zenodo and place them under
   `data/derivatives/nirs-preproc/` at the repository root — see
   `classification/classification_readme.md` for the exact layout.
2. Set up the environment and run the pipeline end-to-end:

   ```bash
   conda env create -f environment.yml
   conda activate respra
   python classification/generate_npy.py
   python classification/publish_riemann_ensemble.py
   ```

Outputs land in `results/publish_ensemble/`.

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
