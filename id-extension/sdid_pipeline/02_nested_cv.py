"""
02_nested_cv.py
Nested cross-validation classification for the main AND the sensitivity
analysis in a single script.

Usage: python 02_nested_cv.py --mode main|sensitivity

  main         full pipeline search (3 feature-extraction x 3 classification methods)
  sensitivity  shared xDAWN latent space with a fixed dimensionality
               (3 classification methods); controls for the difference in
               feature dimensionality between SD and ID

All paths come from config.py. The shared CV code lives in core/cv_core.py.
"""

import warnings
warnings.filterwarnings("ignore")

import config
from core.cv_core import build_pipelines_main, build_pipelines_shared_space, run_nested_cv

MODE = config.resolve_mode()
paths = config.get_mode_paths(MODE)

N_XDAWN_COMPONENTS = 3  # virtual channels per class -> 2 classes x 3 = 6 latent channels per configuration

if MODE == "main":
    pipelines, param_grids = build_pipelines_main()
    block_size_by_condition = config.BLOCK_SIZE_BY_CONDITION  # only used by the block_covariances pipeline
    extra_fields = None
    print("\n[INFO] Mode 'main' (full pipeline search). Expected runtime: several hours for ~100 participants.")
else:
    pipelines, param_grids = build_pipelines_shared_space(n_components=N_XDAWN_COMPONENTS)
    block_size_by_condition = None
    extra_fields = {"latent_dim": 2 * N_XDAWN_COMPONENTS}
    print("\n[INFO] Mode 'sensitivity' (shared xDAWN space). Expected runtime: about 1-2 hours for ~100 participants.")
print("Starting nested CV...\n")

summary_csv, trials_csv = run_nested_cv(
    data_path=paths["data_path"],
    output_dir=paths["output_dir"],
    pipelines=pipelines,
    param_grids=param_grids,
    conditions=config.CONDITIONS,
    random_state=config.RANDOM_STATE,
    block_size_by_condition=block_size_by_condition,
    extra_summary_fields=extra_fields,
    file_prefix=paths["file_prefix"],
)

print(f"[{MODE}] Done.")
print(f"Summary: {summary_csv}")
print(f"Trials:  {trials_csv}")
