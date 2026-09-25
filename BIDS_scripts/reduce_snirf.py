#!/usr/bin/env python3
"""
Reduce the raw SNIRF recordings from the 8x8 to the 5x5 montage.

For every SNIRF file in the input folder the script writes a reduced copy to
the output folder. The original files are never modified. The reduction

  1. keeps only the channels with ``dataTypeLabel == 'RAW'`` in ``nirs/dataN``
     (all other data types are removed, including their ``measurementList``
     entries and ``dataTimeSeries`` columns),
  2. removes the specified source and detector optodes together with every
     channel that uses them,
  3. renumbers sources, detectors and ``measurementList`` entries
     consecutively (1..N), so that no old indices remain,
  4. trims ``probe/sourcePos2D/3D``, ``probe/detectorPos2D/3D`` and, if
     present, ``probe/sourceLabels`` / ``probe/detectorLabels`` accordingly,
  5. rewrites ``metaDataTags/ChannelMask`` to match the new channel count,
  6. copies everything else (``aux*``, ``stim*``, remaining metadata tags, ...)
     unchanged.

By default, sources S2, S4, S5 and detectors D2, D4, D7 are removed (1-based
indices of the original montage). The remaining optodes (S1, S3, S6, S7, S8
and D1, D3, D5, D6, D8) become S1-S5 and D1-D5 of the 5x5 montage. These
indices are also used in ``bidsify_raw.py`` to look up the original optode
labels.

Usage
-----
    python reduce_snirf.py --input INPUT_DIR --output OUTPUT_DIR

Requirements: h5py, numpy (see ``environment.yml``).
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

DEFAULT_SOURCES_TO_REMOVE = [2, 4, 5]
DEFAULT_DETECTORS_TO_REMOVE = [2, 4, 7]
DEFAULT_RAW_LABEL = "RAW"  # dataTypeLabel of the channels that are kept


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================


def build_index_map(n_total, remove_list):
    """Map old (1-based) to new (1-based) indices for the optodes that are kept,
    numbered consecutively without gaps."""
    keep = [i for i in range(1, n_total + 1) if i not in remove_list]
    return {old: new for new, old in enumerate(keep, start=1)}


def read_label(ml_group, field):
    """Read a scalar (string) field of a measurementList group."""
    arr = np.array(ml_group[field][()])
    value = arr.reshape(-1)[0] if arr.size else None
    if isinstance(value, bytes):
        value = value.decode()
    return value


# =============================================================================
# 2. REDUCTION OF ONE FILE
# =============================================================================


def process_data_group(data_grp, src_map, det_map, raw_label, n_wavelengths):
    """Keep only RAW channels of the retained optodes and renumber everything.

    Returns (number of RAW entries before, number of entries after,
    number of channels after per wavelength).
    """
    ml_keys = sorted(
        [k for k in data_grp.keys() if k.startswith("measurementList")],
        key=lambda s: int("".join(filter(str.isdigit, s)) or 0),
    )

    n_raw_before = 0
    surviving = []  # (old key, new source index, new detector index), original order
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

    # Keep only the dataTimeSeries columns of the surviving channels
    dts = data_grp["dataTimeSeries"]
    dts_dtype = dts.dtype
    keep_columns = [ml_keys.index(key) for key, _, _ in surviving]
    new_dts_data = dts[()][:, keep_columns]

    # Build the new measurementList groups in a temporary group ...
    tmp_name = "_tmp_measurementLists"
    if tmp_name in data_grp:
        del data_grp[tmp_name]
    tmp_grp = data_grp.create_group(tmp_name)
    for new_num, (old_key, new_src, new_det) in enumerate(surviving, start=1):
        new_name = f"measurementList{new_num}"
        data_grp.copy(old_key, tmp_grp, name=new_name)
        tmp_grp[new_name]["sourceIndex"][...] = new_src
        tmp_grp[new_name]["detectorIndex"][...] = new_det

    # ... delete ALL old measurementList entries (RAW, OD, Hb) ...
    for key in ml_keys:
        del data_grp[key]

    # ... and write the new, gap-free measurementList1..N back.
    for new_num in range(1, n_after + 1):
        new_name = f"measurementList{new_num}"
        data_grp.copy(tmp_grp[new_name], data_grp, name=new_name)
    del data_grp[tmp_name]

    del data_grp["dataTimeSeries"]
    data_grp.create_dataset("dataTimeSeries", data=new_dts_data, dtype=dts_dtype)

    n_channels_after = n_after // n_wavelengths if n_wavelengths else n_after
    return n_raw_before, n_after, n_channels_after


def trim_probe_array(probe_grp, field_name, keep_indices_1based):
    """Keep only the rows of a probe array that belong to retained optodes."""
    if field_name not in probe_grp:
        return
    ds = probe_grp[field_name]
    dtype = ds.dtype
    new_data = ds[()][[i - 1 for i in keep_indices_1based]]
    del probe_grp[field_name]
    probe_grp.create_dataset(field_name, data=new_data, dtype=dtype)


def update_channel_mask(metadata_grp, n_channels_after):
    """Rewrite ChannelMask (all '1', one entry per channel and wavelength) so
    that no outdated mask from the original montage remains."""
    if "ChannelMask" not in metadata_grp:
        return
    dtype = metadata_grp["ChannelMask"].dtype
    new_mask = ",".join(["1"] * n_channels_after)
    del metadata_grp["ChannelMask"]
    metadata_grp.create_dataset(
        "ChannelMask", data=np.array([new_mask], dtype=object), dtype=dtype
    )


def process_file(in_path, out_path, sources_to_remove, detectors_to_remove, raw_label):
    """Write a reduced copy of one SNIRF file and return a list of report lines."""
    shutil.copy2(in_path, out_path)
    report = []

    with h5py.File(out_path, "r+") as f:
        for nirs_key in [k for k in f.keys() if k.startswith("nirs")]:
            nirs = f[nirs_key]
            probe = nirs["probe"]

            n_src_total = probe["sourcePos2D"].shape[0]
            n_det_total = probe["detectorPos2D"].shape[0]
            n_wavelengths = probe["wavelengths"].shape[0] if "wavelengths" in probe else 1

            src_map = build_index_map(n_src_total, sources_to_remove)
            det_map = build_index_map(n_det_total, detectors_to_remove)
            keep_src = sorted(src_map)
            keep_det = sorted(det_map)

            for field in ("sourcePos2D", "sourcePos3D", "sourceLabels"):
                trim_probe_array(probe, field, keep_src)
            for field in ("detectorPos2D", "detectorPos3D", "detectorLabels"):
                trim_probe_array(probe, field, keep_det)
            report.append(
                f"    {nirs_key}/probe: sources {n_src_total} -> {len(keep_src)}, "
                f"detectors {n_det_total} -> {len(keep_det)}"
            )

            n_channels_after = None
            for data_key in [k for k in nirs.keys() if k.startswith("data")]:
                n_raw_before, n_after, n_channels_after = process_data_group(
                    nirs[data_key], src_map, det_map, raw_label, n_wavelengths
                )
                report.append(
                    f"    {nirs_key}/{data_key}: {n_raw_before} RAW entries -> {n_after} "
                    f"({n_channels_after} channels x {n_wavelengths} wavelengths); "
                    f"OD/Hb entries removed"
                )

            if n_channels_after is not None and "metaDataTags" in nirs:
                update_channel_mask(nirs["metaDataTags"], n_channels_after)

    return report


# =============================================================================
# 3. MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce SNIRF recordings from the 8x8 to the 5x5 montage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="folder with the original .snirf files")
    parser.add_argument("--output", type=Path, required=True,
                        help="folder for the reduced .snirf files")
    parser.add_argument("--remove-sources", type=int, nargs="+",
                        default=DEFAULT_SOURCES_TO_REMOVE,
                        help="sources to remove (1-based)")
    parser.add_argument("--remove-detectors", type=int, nargs="+",
                        default=DEFAULT_DETECTORS_TO_REMOVE,
                        help="detectors to remove (1-based)")
    parser.add_argument("--raw-label", default=DEFAULT_RAW_LABEL,
                        help="dataTypeLabel of the channels to keep")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.is_dir():
        print(f"ERROR: input folder not found: {args.input}")
        return
    files = sorted(args.input.glob("*.snirf"))
    if not files:
        print(f"ERROR: no .snirf files found in {args.input}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files found.")
    print(f"Removing sources {args.remove_sources} and detectors {args.remove_detectors}.")
    print(f"Keeping only dataTypeLabel == '{args.raw_label}'.")
    print(f"Output folder: {args.output}\n")

    failed = []
    for i, in_path in enumerate(files, start=1):
        out_path = args.output / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name}")
        try:
            for line in process_file(in_path, out_path, args.remove_sources,
                                     args.remove_detectors, args.raw_label):
                print(line)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(in_path.name)
            if out_path.exists():
                out_path.unlink()

    print("\nDone.")
    print(f"Successfully processed: {len(files) - len(failed)}/{len(files)}")
    if failed:
        print("Failed files:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
