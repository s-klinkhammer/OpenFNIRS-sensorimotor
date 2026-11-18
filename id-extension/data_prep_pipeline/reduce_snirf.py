#!/usr/bin/env python3
"""
reduce_snirf.py - Reduce raw SNIRF files to a chosen optode setup
=================================================================

Creates reduced copies of SNIRF files in ONE pass:

  1. Keeps only the channels with dataTypeLabel == 'RAW' in nirs/dataN
     (OD and HbO/HbR entries are removed completely, including their
     measurementList entries and dataTimeSeries columns).
  2. Removes the sources and detectors listed for the selected setup
     (including every channel that uses them).
  3. Renumbers sources, detectors and measurementList entries gaplessly
     (1..N), so no old indices are left over.
  4. Trims probe/sourcePos2D/3D, detectorPos2D/3D and, if present,
     sourceLabels/detectorLabels accordingly.
  5. Rewrites metaDataTags/ChannelMask to the new channel count
     (otherwise a wrong/outdated mask would remain).
  6. Copies all other groups (aux*, stim*, remaining metaDataTags,
     formatVersion, landmarks, ...) unchanged.

The original files are NEVER modified; copies are written to the output
folder.

Setups
------
The input files must contain the FULL 8x8 recording (original optode
numbering). You choose which setup to create from it:

  ID_8x8  Increased Density: full 8x8 montage, no optodes are removed.
  SD_5x5  Standard Density: reduced 5x5 montage, sources S2, S4, S5 and
          detectors D2, D4, D7 are removed.

Both setups also drop all non-RAW data (OD, HbO/HbR).

How to run
----------
Open a terminal in the folder containing this script and run ONE of:

  python reduce_snirf.py --input <input folder> --output <output folder> --setup ID_8x8
  python reduce_snirf.py --input <input folder> --output <output folder> --setup SD_5x5

Example (SD):
  python reduce_snirf.py --input data/raw_8x8 --output data/SD_5x5 --setup SD_5x5

Instead of typing the options each time, you can fill in INPUT_FOLDER,
OUTPUT_FOLDER and SETUP in the configuration block below. Command-line
options take precedence. Run "python reduce_snirf.py --help" for a summary.

Requirements: h5py, numpy
    pip install h5py numpy
"""

import argparse
import glob
import os
import shutil
import sys

import h5py
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths and setup can be set here or passed on the command line.
# Leave as None to require them on the command line.
INPUT_FOLDER = None     # folder containing the .snirf files to reduce
OUTPUT_FOLDER = None    # folder where the reduced copies are written
SETUP = None            # key of SETUPS below, e.g. "SD_5x5"

# Optodes to remove per setup. Indices are 1-based and refer to the ORIGINAL
# numbering of the input files (e.g. [2, 4, 5] means S2, S4, S5).
SETUPS = {
    # Full increased-density setup: nothing is removed
    "ID_8x8": {"sources": [], "detectors": []},
    # Standard-density subset: remove S2, S4, S5 and D2, D4, D7
    "SD_5x5": {"sources": [2, 4, 5], "detectors": [2, 4, 7]},
}

RAW_LABEL = "RAW"       # dataTypeLabel value that is kept

# =============================================================================


def build_index_map(n_total, remove_list):
    """Map old 1-based index -> new 1-based index for the surviving
    optodes, numbered upwards without gaps."""
    keep = [i for i in range(1, n_total + 1) if i not in remove_list]
    return {old: new for new, old in enumerate(keep, start=1)}


def read_label(ml_group, field):
    """Read a scalar field of a measurementList group (decode bytes)."""
    val = ml_group[field][()]
    arr = np.array(val)
    v = arr.reshape(-1)[0] if arr.size else None
    if isinstance(v, bytes):
        v = v.decode()
    return v


def process_data_group(data_grp, src_map, det_map, raw_label, n_wavelengths):
    """Keep only RAW channels, drop removed optodes and renumber everything.

    Returns (n_raw_before, n_entries_after, n_channels_after), where
    n_channels_after is the number of channels per wavelength."""

    ml_keys = sorted(
        [k for k in data_grp.keys() if k.startswith("measurementList")],
        key=lambda s: int(''.join(filter(str.isdigit, s)) or 0)
    )

    n_raw_before = 0
    surviving = []  # (old_key, new_src, new_det), in original order

    for key in ml_keys:
        ml = data_grp[key]
        label = read_label(ml, "dataTypeLabel")
        if label is None or label.strip().upper() != raw_label.upper():
            continue
        n_raw_before += 1
        old_src = int(np.array(ml["sourceIndex"][()]).item())
        old_det = int(np.array(ml["detectorIndex"][()]).item())
        if old_src in src_map and old_det in det_map:
            surviving.append((key, src_map[old_src], det_map[old_det]))

    n_after = len(surviving)

    # --- trim dataTimeSeries to the surviving columns ---
    dts = data_grp["dataTimeSeries"]
    dts_data = dts[()]
    dts_dtype = dts.dtype
    keep_col_positions = [ml_keys.index(k) for (k, _, _) in surviving]
    new_dts_data = dts_data[:, keep_col_positions]

    # --- build new measurementList groups (via a temporary group) ---
    tmp_name = "_tmp_measurementLists"
    if tmp_name in data_grp:
        del data_grp[tmp_name]
    tmp_grp = data_grp.create_group(tmp_name)

    for new_num, (old_key, new_src, new_det) in enumerate(surviving, start=1):
        new_name = f"measurementList{new_num}"
        data_grp.copy(old_key, tmp_grp, name=new_name)
        tmp_grp[new_name]["sourceIndex"][...] = new_src
        tmp_grp[new_name]["detectorIndex"][...] = new_det

    # delete ALL old measurementList entries (RAW + OD + Hb)
    for key in ml_keys:
        del data_grp[key]

    # write the new, gapless measurementList1..N back
    for new_num in range(1, n_after + 1):
        new_name = f"measurementList{new_num}"
        data_grp.copy(tmp_grp[new_name], data_grp, name=new_name)
    del data_grp[tmp_name]

    # replace dataTimeSeries
    del data_grp["dataTimeSeries"]
    data_grp.create_dataset("dataTimeSeries", data=new_dts_data, dtype=dts_dtype)

    n_channels_after = n_after // n_wavelengths if n_wavelengths else n_after
    return n_raw_before, n_after, n_channels_after


def trim_probe_array(probe_grp, field_name, keep_indices_1based):
    """Keep only the given 1-based rows of a probe dataset (if present)."""
    if field_name not in probe_grp:
        return None, None
    ds = probe_grp[field_name]
    data = ds[()]
    dtype = ds.dtype
    zero_based = [i - 1 for i in keep_indices_1based]
    new_data = data[zero_based]
    del probe_grp[field_name]
    probe_grp.create_dataset(field_name, data=new_data, dtype=dtype)
    return data.shape[0], new_data.shape[0]


def update_channel_mask(metadata_grp, n_channels_after):
    """Rewrite ChannelMask (all '1', length = new channel count per
    wavelength) so that no outdated mask remains."""
    if "ChannelMask" not in metadata_grp:
        return
    ds = metadata_grp["ChannelMask"]
    dtype = ds.dtype
    new_mask = ",".join(["1"] * n_channels_after)
    del metadata_grp["ChannelMask"]
    metadata_grp.create_dataset(
        "ChannelMask", data=np.array([new_mask], dtype=object), dtype=dtype
    )


def process_file(in_path, out_path, sources_to_remove, detectors_to_remove):
    """Copy one file and reduce the copy. Returns a list of report lines."""
    shutil.copy2(in_path, out_path)
    report = []

    with h5py.File(out_path, "r+") as f:
        nirs_keys = [k for k in f.keys() if k.startswith("nirs")]

        for nk in nirs_keys:
            nirs = f[nk]
            probe = nirs["probe"]

            n_src_total = probe["sourcePos2D"].shape[0]
            n_det_total = probe["detectorPos2D"].shape[0]
            n_wavelengths = (
                probe["wavelengths"].shape[0] if "wavelengths" in probe else 1
            )

            # fail early if the setup does not fit the file
            bad_src = [i for i in sources_to_remove if not 1 <= i <= n_src_total]
            bad_det = [i for i in detectors_to_remove if not 1 <= i <= n_det_total]
            if bad_src or bad_det:
                raise ValueError(
                    f"Setup does not match file: sources {bad_src} / detectors "
                    f"{bad_det} out of range (file has {n_src_total} sources, "
                    f"{n_det_total} detectors)"
                )

            src_map = build_index_map(n_src_total, sources_to_remove)
            det_map = build_index_map(n_det_total, detectors_to_remove)
            keep_src = sorted(src_map.keys())
            keep_det = sorted(det_map.keys())

            for field in ["sourcePos2D", "sourcePos3D", "sourceLabels"]:
                trim_probe_array(probe, field, keep_src)
            for field in ["detectorPos2D", "detectorPos3D", "detectorLabels"]:
                trim_probe_array(probe, field, keep_det)

            report.append(
                f"    {nk}/probe: sources {n_src_total} -> {len(keep_src)}, "
                f"detectors {n_det_total} -> {len(keep_det)}"
            )

            data_keys = [k for k in nirs.keys() if k.startswith("data")]
            n_channels_after = None
            for dk in data_keys:
                n_raw_before, n_after, n_channels_after = process_data_group(
                    nirs[dk], src_map, det_map, RAW_LABEL, n_wavelengths
                )
                report.append(
                    f"    {nk}/{dk}: {n_raw_before} RAW entries -> {n_after} "
                    f"({n_channels_after} channels x {n_wavelengths} wavelengths); "
                    f"OD/Hb entries removed completely"
                )

            if n_channels_after is not None and "metaDataTags" in nirs:
                update_channel_mask(nirs["metaDataTags"], n_channels_after)

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce raw SNIRF files to a chosen optode setup."
    )
    parser.add_argument("--input", default=INPUT_FOLDER,
                        help="folder with the input .snirf files")
    parser.add_argument("--output", default=OUTPUT_FOLDER,
                        help="folder for the reduced copies")
    parser.add_argument("--setup", default=SETUP, choices=sorted(SETUPS),
                        help="setup to produce (key of SETUPS in the script)")
    return parser.parse_args()


def main():
    args = parse_args()

    missing = [name for name, val in
               (("--input", args.input), ("--output", args.output),
                ("--setup", args.setup)) if not val]
    if missing:
        print(f"ERROR: missing {', '.join(missing)} "
              f"(set in the config block or pass on the command line).")
        sys.exit(1)

    input_folder = args.input
    output_folder = args.output
    sources_to_remove = SETUPS[args.setup]["sources"]
    detectors_to_remove = SETUPS[args.setup]["detectors"]

    if not os.path.isdir(input_folder):
        print(f"ERROR: input folder not found: {input_folder}")
        sys.exit(1)

    if os.path.abspath(input_folder) == os.path.abspath(output_folder):
        print("ERROR: input and output folder must differ "
              "(originals are never overwritten).")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_folder, "*.snirf")))
    if not files:
        print(f"ERROR: no .snirf files found in {input_folder}")
        sys.exit(1)

    print(f"{len(files)} files found.")
    print(f"Setup: {args.setup}")
    print(f"Removing sources {sources_to_remove} and detectors {detectors_to_remove}.")
    print(f"Keeping only dataTypeLabel == '{RAW_LABEL}'.")
    print(f"Output folder: {output_folder}\n")

    failed = []
    for i, in_path in enumerate(files, start=1):
        fname = os.path.basename(in_path)
        out_path = os.path.join(output_folder, fname)
        print(f"[{i}/{len(files)}] {fname}")
        try:
            for line in process_file(in_path, out_path,
                                     sources_to_remove, detectors_to_remove):
                print(line)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(fname)
            if os.path.exists(out_path):
                os.remove(out_path)

    print("\nDone.")
    print(f"Successfully processed: {len(files) - len(failed)}/{len(files)}")
    if failed:
        print("Failed files:")
        for fn in failed:
            print(f"  - {fn}")
        sys.exit(1)


if __name__ == "__main__":
    main()
