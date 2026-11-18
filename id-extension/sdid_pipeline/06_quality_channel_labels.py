"""
06_quality_channel_labels.py
Per-participant signal-quality labels (Good / Moderate / Poor).

The label is based only on the NUMBER of rejected ID channels per run (not on
channel numbers), so ID and SD channel numbering cannot be confused:
  Good     : 0-1 rejected channels (<= 5 % of 20)
  Moderate : 2-3 rejected channels
  Poor     : >= 4 rejected channels (>= 20 % of 20)
The combined label of a participant is the worse of the two runs.

Note on channel identity: SD is a subset of ID, but the input files number the
channels separately (ID: 1-20, SD: 1-13). Channel numbers must therefore NOT be
compared directly between ID and SD. Overlap columns are only computed if the
mapping SD_TO_ID below is filled in.

Inputs : rejectedChannels_ID_*.xlsx / rejectedChannels_SD_*.xlsx from step 05
Usage  : python 06_quality_channel_labels.py [--id-xlsx FILE] [--sd-xlsx FILE]
"""

import os
import argparse
from datetime import datetime

import pandas as pd

import config

ID_TOTAL = 20
SD_TOTAL = 13

# Thresholds (number of rejected ID channels per run)
GOOD_MAX_REJECTED = 1
POOR_MIN_REJECTED = 4

# Optional mapping SD channel number -> ID channel number (physical channel),
# e.g. {1: 2, 2: 5, ...} with 13 entries. While None, no overlap columns are written.
SD_TO_ID = None

LABEL_ORDER = {"Good": 0, "Moderate": 1, "Poor": 2}


def run_label(n_rejected_id):
    if n_rejected_id <= GOOD_MAX_REJECTED:
        return "Good"
    if n_rejected_id < POOR_MIN_REJECTED:
        return "Moderate"
    return "Poor"


def parse_channels(x):
    """'3,7,12' -> {3, 7, 12}; '0' or '' -> empty set."""
    if x == "0" or x == "":
        return set()
    return set(int(i) for i in str(x).split(","))


parser = argparse.ArgumentParser(description="Channel quality labels (Good/Moderate/Poor)")
parser.add_argument("--id-xlsx", help="rejectedChannels_ID_*.xlsx (default: latest in RESULTS_QUALITY_DIR)")
parser.add_argument("--sd-xlsx", help="rejectedChannels_SD_*.xlsx (default: latest in RESULTS_QUALITY_DIR)")
args = parser.parse_args()
config.ensure_output_dirs()
id_path = args.id_xlsx or config.find_latest(config.RESULTS_QUALITY_DIR, "rejectedChannels_ID", ".xlsx")
sd_path = args.sd_xlsx or config.find_latest(config.RESULTS_QUALITY_DIR, "rejectedChannels_SD", ".xlsx")
print(f"ID: {id_path}\nSD: {sd_path}")

id_raw = pd.read_excel(id_path, dtype=str)
sd_raw = pd.read_excel(sd_path, dtype=str)

# Make missing runs visible (otherwise they would silently count as "0 rejected" = Good)
for name, df_ in [("ID", id_raw), ("SD", sd_raw)]:
    for col in ["RunA_count", "RunB_count"]:
        if col not in df_.columns:
            raise ValueError(f"{name}: column {col} is missing.")
    missing = df_[df_[["RunA_count", "RunB_count"]].isna().any(axis=1)]["Participant"].tolist()
    if missing:
        print(f"WARNING ({name}): missing run for {missing}")

id_df = id_raw.fillna("0")
sd_df = sd_raw.fillna("0")

rows = []
for _, id_row in id_df.iterrows():
    participant = id_row["Participant"]
    sd_match = sd_df[sd_df["Participant"] == participant]
    if sd_match.empty:
        print(f"WARNING: {participant} not found in the SD file - skipped.")
        continue
    sd_row = sd_match.iloc[0]

    for run in ["A", "B"]:
        id_rej_n = int(id_row[f"Run{run}_count"])
        id_rej_ch = parse_channels(id_row[f"Run{run}_channels"])
        sd_rej_n = int(sd_row[f"Run{run}_count"])
        sd_rej_ch = parse_channels(sd_row[f"Run{run}_channels"])
        id_left = ID_TOTAL - id_rej_n
        sd_left = SD_TOTAL - sd_rej_n

        row = {
            "Participant": participant, "Run": run,
            "ID_total": ID_TOTAL, "ID_rejected": id_rej_n, "ID_left": id_left,
            "ID_rejected_%": round(id_rej_n / ID_TOTAL * 100, 1),
            "ID_left_%": round(id_left / ID_TOTAL * 100, 1),
            "SD_total": SD_TOTAL, "SD_rejected": sd_rej_n, "SD_left": sd_left,
            "SD_left_%": round(sd_left / SD_TOTAL * 100, 1),
            "ID_rejected_channels": ",".join(map(str, sorted(id_rej_ch))),
            "SD_rejected_channels": ",".join(map(str, sorted(sd_rej_ch))),
        }

        # Overlap only with an explicit channel mapping, so channel identities are preserved
        if SD_TO_ID is not None:
            sd_as_id = {SD_TO_ID[c] for c in sd_rej_ch}   # KeyError = incomplete mapping
            row["Rej_common"] = len(id_rej_ch & sd_as_id)
            row["ID_only_rej"] = len(id_rej_ch - sd_as_id)
            row["SD_only_rej"] = len(sd_as_id - id_rej_ch)  # should be ~0 if SD is a subset of ID

        row["Quality_Label"] = run_label(id_rej_n)
        rows.append(row)

overview_df = pd.DataFrame(rows)

# Combined label = worst label of the two runs
combined_quality = {
    participant: max(group["Quality_Label"], key=lambda l: LABEL_ORDER[l])
    for participant, group in overview_df.groupby("Participant")
}
overview_df["Combined_Quality_Label"] = overview_df["Participant"].map(combined_quality)

counts = pd.Series(combined_quality).value_counts().reindex(["Good", "Moderate", "Poor"]).fillna(0).astype(int)
print("\nParticipants per group:")
print(counts.to_string())
print(f"Total: {counts.sum()}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
save_path = os.path.join(config.RESULTS_QUALITY_DIR, f"quality_overview_{timestamp}.xlsx")
overview_df.to_excel(save_path, index=False)
print(f"Quality overview saved: {save_path}")
