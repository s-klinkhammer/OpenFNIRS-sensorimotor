"""
Recursively searches ROOT for multi-study GLM result files (regardless
of which sub-XXX folder Satori happened to save them in) and moves them
into a central group/ folder.

Matched:
  - MultiStudy_GLM_Results.txt
  - MultiStudy_GLM_Results_MSGLM_Results.xlsx
  - *_MSGLM_*.cmp   (e.g. MSGLM_RFX_t-test.cmp, MSGLM_SPSB_t-test.cmp)

The large design matrix files (MultiStudy_DesignMatrix*) are
intentionally NOT moved, since they are just Satori inputs.
"""

import re
import shutil
from pathlib import Path

# Resolved relative to this script's location, so it works no matter
# which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent / "data" / "derivatives" / "nirs-preproc"
GROUP_DIR = ROOT / "group"

PATTERNS = [
    "MultiStudy_GLM_Results*",
    "*_MSGLM_*.cmp",
]

# The .cmp files come out named like
# "sub-001_task-motor_run-1_desc-preproc_nirs_MSGLM_SPSB_t-test.cmp",
# which looks subject- and run-specific even though the GLM is pooled
# across all subjects and both runs. That whole leading part gets
# replaced with "group_".
SUB_RUN_PREFIX = re.compile(r"^sub-\d+_task-motor_run-\d+_desc-preproc_nirs_")


def target_name(filename: str) -> str:
    return SUB_RUN_PREFIX.sub("group_", filename)


def main():
    if not ROOT.exists():
        print(f"ROOT not found: {ROOT}")
        print("Adjust the ROOT path at the top of the script if your folder layout differs.")
        return

    GROUP_DIR.mkdir(exist_ok=True)
    moved = []

    for pattern in PATTERNS:
        for file in ROOT.rglob(pattern):
            if file.parent == GROUP_DIR:
                continue  # already there

            new_name = target_name(file.name)
            dest = GROUP_DIR / new_name
            if dest.exists():
                print(f"Skipped (already exists in group/): {new_name}")
                continue

            shutil.move(str(file), str(dest))
            if new_name != file.name:
                moved.append(f"{file.name}  ->  {new_name}")
            else:
                moved.append(file.name)

    print(f"\nMoved {len(moved)} file(s) to {GROUP_DIR}:")
    for name in moved:
        print(f"  - {name}")

    if not moved:
        print("No matching files found. Is ROOT set correctly?")


if __name__ == "__main__":
    main()
