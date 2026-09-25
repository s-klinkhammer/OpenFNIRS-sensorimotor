#!/usr/bin/env python3
"""
Convert the raw fNIRS recordings of the OpenFNIRS-sensorimotor dataset to BIDS.

For every included participant the script
  * copies the two SNIRF runs to ``sub-XXX/nirs/`` using BIDS file names,
  * writes the run sidecars (``*_nirs.json``), ``channels.tsv``,
    ``optodes.tsv``, ``coordsystem.json`` and ``events.tsv``,
  * writes ``scans.tsv`` (acquisition time and Aurora software version),
and, at dataset level, ``README``, ``dataset_description.json``,
``participants.tsv`` and ``participants.json``.

Input
-----
--originals     Folder with the original NIRx exports. It is searched
                recursively for ``*probeInfo.mat`` files, which provide the
                optode labels.
--snirf         Folder with the SNIRF files (already reduced to the 5x5
                montage). File names must start with the participant ID
                (e.g. ``P012_...``) and contain the run (``run1``/``runA``/
                ``run_1`` for the first run; everything else is run 2).
--demographics  Excel file with age, sex and handedness per participant.
--exclude       Excel file with an ``exclude`` column containing participants 
                that did not agree to data sharing; every participant with a
                non-empty entry is left out.
--output        Target folder (BIDS root).

Both Excel files need an ID column named ``participant_id``, ``id`` or
``subject`` (otherwise the first column is used).

Usage
-----
    python bidsify_raw.py --originals ORIGINALS_DIR --snirf SNIRF_DIR \\
        --demographics demographics.xlsx --exclude exclude.xlsx \\
        --output BIDS_DIR

Requirements: pandas, openpyxl, h5py, scipy (see ``environment.yml``).
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import pandas as pd
from scipy.io import loadmat

# =============================================================================
# 1. DATASET-LEVEL METADATA
# =============================================================================

README_CONTENT = """# NIRS sensorimotor tasks: Hand clenching & Finger tapping

This dataset contains fNIRS data of 100 subjects performing two sensorimotor tasks.
Tasks: Hand clenching and finger tapping.
Hand clenching = repetitive rhythmic fist closure and release.
Finger tapping = sequential tapping of the thumb to index, middle, ring and pinkie finger.

Data were collected across two identical runs without removing the cap in between.
Each run followed an alternating block design (5 repetitions per task) consisting of
10 s task periods (hand clenching, finger tapping) and 20 s rest periods (plus 2 s jitter).
This resulted in 10 trials per run and a total of 20 trials per participant.

Device: NIRx NIRSport2
Cap: EasyCap (10-5 System)


## Derivatives

This dataset includes preprocessed fNIRS data located in the `derivatives/nirs-preproc/` directory.

The preprocessing was performed using Satori (v2.2.4).
The pipeline includes:
* Optical Density conversion and Hb concentration calculation (mBLL).
* Artifact correction (Spike Removal & TDDR).
* Short-channel regression and temporal filtering (0.01 - 0.09 Hz).
* Z-transformation and epoching.

For a full, detailed description of the parameters and the complete pipeline, please refer to the `README` file located within `derivatives/nirs-preproc/`.

## Contact
For questions, contact: Bettina Sorger at Maastricht University (b.sorger@maastrichtuniversity.nl).
"""

DATASET_DESCRIPTION = {
    "Name": "fNIRS sensorimotor tasks: Hand clenching & Finger tapping",
    "BIDSVersion": "1.11.1",
    "DatasetType": "raw",
    "License": "CC BY",
    "Authors": [
        "Simona Klinkhammer",
        "Tim Naeher",
        "Sophie Raible",
        "Michael Luehrs",
        "Franziska Klein",
        "Bettina Sorger",
    ],
    "Keywords": [
        "functional near-infrared spectroscopy",
        "open-access dataset",
        "machine learning",
        "motor execution",
        "sensorimotor",
        "brain-computer Interface",
    ],
    "HowToAcknowledge": "Please cite this paper: [insert link]",
    "Funding": [
        "This work was funded through The Netherlands Organization for "
        "Scientific Research (NWO; Vidi-Grant No. VI.Vidi.191.210 to Bettina Sorger)"
    ],
    "GeneratedBy": [
        {
            "Name": "Custom BIDSification script (Python)",
            "Description": (
                "Custom in-house pipeline converting raw fNIRS .snirf recordings "
                "and .tri trigger files into BIDS format, including conversion of "
                "the original 8x8 optode montage to a reduced 5x5 configuration "
                "and event extraction from SNIRF stim data."
            ),
        }
    ],
}

PARTICIPANTS_METADATA = {
    "participant_id": {"Description": "Unique participant identifier"},
    "age": {"Description": "Age of the participant", "Units": "years"},
    "sex": {"Description": "Biological sex", "Levels": {"f": "female", "m": "male"}},
    "handedness": {"Description": "Handedness", "Levels": {"right": "right", "left": "left"}},
}

# =============================================================================
# 2. RECORDING-LEVEL METADATA
# =============================================================================

# Sidecar shared by all runs (written to ``*_task-motor_run-<n>_nirs.json``).
NIRS_SIDECAR = {
    "TaskName": "motor",
    "Manufacturer": "NIRx",
    "ManufacturersModelName": "NIRSport2",
    "SamplingFrequency": 10.1725,
    "NIRSChannelCount": 32,
    "NIRSSourceOptodeCount": 5,
    "NIRSDetectorOptodeCount": 5,
    "ACCELChannelCount": 3,
    "GYROChannelCount": 3,
    "CapManufacturer": "EasyCap",
    "SourceType": "LED",
    "DetectorType": "SiPD",
    "ShortChannelCount": 2,
    "NIRSPlacementScheme": "10-5",
    "SoftwareVersions": "Aurora version(s) can be found in individual scan.tsv files",
    "InstitutionName": "Maastricht University",
    "InstitutionalDepartmentName": (
        "Department of Cognitive Neuroscience at the Faculty of Psychology and Neuroscience"
    ),
}

TASK_DURATION = 10.0  # duration of each task block in seconds

# Mapping of the stim names in the SNIRF files to task labels. To swap the
# labels, this is the only place that needs to be changed.
TRIGGER_LABELS = {
    "1": "hand_clenching",
    "2": "finger_tapping",
}

WAVELENGTHS = {1: 760, 2: 850}  # SNIRF wavelength index -> nominal wavelength (nm)

# Montage reduction: the recordings were made with an 8x8 montage and reduced
# to 5x5. These are the original (1-based) indices of the retained optodes;
# they are renumbered 1..5 in the reduced files.
ORIGINAL_SOURCES = [1, 3, 6, 7, 8]
ORIGINAL_DETECTORS = [1, 3, 5, 6, 8]

# Short-distance channel in the renumbered 5x5 layout: (source, detector).
SHORT_CHANNEL = (5, 2)

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)


def write_tsv(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def digits(text):
    """Return only the digits of a string (e.g. 'P012' -> '012')."""
    return re.sub(r"\D", "", str(text))


def as_str(value):
    """Decode a value read from an HDF5 file to a plain string."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_table(path):
    """Read an Excel file and return the table and the name of its ID column."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = next(
        (c for c in df.columns if c in ("participant_id", "id", "subject")),
        df.columns[0],
    )
    return df, id_col


def load_excluded_ids(path):
    """Return the digit-only IDs of all participants flagged in the 'exclude' column."""
    df, id_col = read_table(path)
    if "exclude" not in df.columns:
        return []
    return [digits(x) for x in df.loc[df["exclude"].notna(), id_col].astype(str)]


def detect_run(filename):
    """Return the run number ('1' or '2') encoded in a SNIRF file name."""
    name = filename.lower()
    return "1" if any(tag in name for tag in ("run1", "runa", "run_1")) else "2"


def read_scan_info(snirf_path):
    """Read acquisition time and Aurora version from the SNIRF metadata tags."""
    acq_time = "n/a"
    aurora_version = "n/a"
    try:
        with h5py.File(snirf_path, "r") as f:
            meta = f.get("nirs/metaDataTags")
            if meta is not None:
                if "MeasurementDate" in meta and "MeasurementTime" in meta:
                    date = as_str(meta["MeasurementDate"][0])
                    time = as_str(meta["MeasurementTime"][0])
                    acq_time = f"{date}T{time}"
                if "AuroraVersion" in meta:
                    parts = as_str(meta["AuroraVersion"][0]).split("-")
                    short_version = "-".join(parts[:2]) if len(parts) > 1 else parts[0]
                    aurora_version = f"Aurora {short_version}"
    except Exception as e:
        print(f"WARNING {snirf_path.name}: could not read metadata tags ({e})")
    return acq_time, aurora_version


# =============================================================================
# 4. EVENTS
# =============================================================================


def _decode_stim_name(raw_name):
    """Convert the raw HDF5 'name' field of a stim group to a plain string."""
    if isinstance(raw_name, bytes):
        return raw_name.decode()
    if hasattr(raw_name, "__len__") and not isinstance(raw_name, str):
        first = raw_name[0] if len(raw_name) else raw_name
        return _decode_stim_name(first)
    return str(raw_name)


def write_events(dest, bid):
    """Write one ``events.tsv`` per run, derived from the stim groups in the SNIRF file.

    Events are taken from the SNIRF files rather than from the raw .tri
    trigger codes: the stim names in the SNIRF files are normalised (stim '1'
    always occurs first and is hand clenching, stim '2' follows and is finger
    tapping), whereas the raw hardware trigger codes differed between
    participants depending on the stimulation script that was used.

    Runs that do not follow the expected pattern (exactly two stim groups,
    '1' before '2') are skipped with a warning instead of being labelled
    possibly incorrectly.
    """
    for snirf_path in sorted(dest.glob(f"{bid}_task-motor_run-*_nirs.snirf")):
        run = snirf_path.stem.split("run-")[1].split("_")[0]

        stims = []  # list of (stim name, array of onsets)
        try:
            with h5py.File(snirf_path, "r") as f:
                nirs_group = f["nirs"]
                i = 1
                while f"stim{i}" in nirs_group:
                    stim_group = nirs_group[f"stim{i}"]
                    name = _decode_stim_name(stim_group["name"][()])
                    data = stim_group["data"][()]
                    stims.append((name, data[:, 0]))  # first column = onset
                    i += 1
        except Exception as e:
            print(f"WARNING {snirf_path.name}: could not read stim data ({e}) - skipped!")
            continue

        if len(stims) != 2:
            print(f"WARNING {snirf_path.name}: {len(stims)} stim groups instead of 2 "
                  f"- skipped, please check manually!")
            continue

        first_group, second_group = sorted(stims, key=lambda s: s[1].min())
        if first_group[0] != "1" or second_group[0] != "2":
            print(f"WARNING {snirf_path.name}: unexpected order/codes "
                  f"('{first_group[0]}' before '{second_group[0]}', expected '1' before '2') "
                  f"- skipped, please check manually!")
            continue

        events = [
            {
                "onset": round(float(onset), 4),
                "duration": TASK_DURATION,
                "trial_type": TRIGGER_LABELS.get(name, "unknown"),
                "value": name,
            }
            for name, onsets in stims
            for onset in onsets
        ]
        events = pd.DataFrame(events).sort_values("onset").reset_index(drop=True)
        events.to_csv(dest / f"{bid}_task-motor_run-{run}_events.tsv", sep="\t", index=False)


# =============================================================================
# 5. CHANNELS, OPTODES AND COORDINATE SYSTEM
# =============================================================================


def build_channel_table(snirf):
    """Build the rows of channels.tsv: fNIRS channels (from the SNIRF measurement
    list) followed by the three accelerometer and three gyroscope axes."""
    measurement_lists = snirf["nirs/data1"]
    combos = []  # (source, detector, wavelength)
    for key in measurement_lists.keys():
        if not key.startswith("measurementList"):
            continue
        ml = measurement_lists[key]
        combos.append((
            int(ml["sourceIndex"][0]),
            int(ml["detectorIndex"][0]),
            WAVELENGTHS.get(int(ml["wavelengthIndex"][0]), 850),
        ))

    rows = []
    for s, d, wl in sorted(combos):
        rows.append({
            "name": f"S{s}-D{d}_{wl}",
            "type": "NIRSCWAMPLITUDE",
            "source": f"S{s}",
            "detector": f"D{d}",
            "wavelength_nominal": wl,
            "units": "V",
            "short_channel": "true" if (s, d) == SHORT_CHANNEL else "false",
            "component": "n/a",
        })

    for label, unit in (("ACCEL", "m/s^2"), ("GYRO", "rad/s")):
        for axis in ("x", "y", "z"):
            rows.append({
                "name": f"{label}_{axis.upper()}",
                "type": label,
                "source": "n/a",
                "detector": "n/a",
                "wavelength_nominal": "n/a",
                "units": unit,
                "short_channel": "false",
                "component": axis,
            })
    return rows


def build_optode_table(snirf, probe_info):
    """Build the rows of optodes.tsv.

    Template positions come from the (already reduced) SNIRF probe. The
    description holds the optode label of the original 8x8 montage, looked up
    in probeInfo.mat via the original optode index.
    """
    rows = []
    groups = (
        ("S", "source", snirf["nirs/probe/sourcePos3D"][:],
         ORIGINAL_SOURCES, probe_info.probes.labels_s),
        ("D", "detector", snirf["nirs/probe/detectorPos3D"][:],
         ORIGINAL_DETECTORS, probe_info.probes.labels_d),
    )
    for prefix, kind, positions, original_indices, labels in groups:
        for new_idx, original_idx in enumerate(original_indices):
            mat_idx = original_idx - 1  # 0-based index in probeInfo.mat
            rows.append({
                "name": f"{prefix}{new_idx + 1}",
                "type": kind,
                "x": "n/a",
                "y": "n/a",
                "z": "n/a",
                "template_x": round(positions[new_idx][0], 6),
                "template_y": round(positions[new_idx][1], 6),
                "template_z": round(positions[new_idx][2], 6),
                "description": labels[mat_idx] if mat_idx < len(labels) else "[]",
            })
    return rows


def write_hardware_files(originals_dir, snirf_path, file_id, bid, dest):
    """Write channels.tsv, optodes.tsv and coordsystem.json for one participant."""
    probe_files = list(originals_dir.rglob(f"*{file_id}*probeInfo.mat"))
    if not probe_files:  # fall back to any probeInfo file
        probe_files = list(originals_dir.rglob("*probeInfo.mat"))
    if not probe_files:
        print(f"WARNING {bid}: no probeInfo.mat found - hardware files not written")
        return

    try:
        probe_info = loadmat(probe_files[0], struct_as_record=False, squeeze_me=True)["probeInfo"]
        with h5py.File(snirf_path, "r") as snirf:
            write_tsv(dest / f"{bid}_task-motor_channels.tsv", build_channel_table(snirf))
            write_tsv(dest / f"{bid}_optodes.tsv", build_optode_table(snirf, probe_info))
        write_json(dest / f"{bid}_coordsystem.json", {
            "NIRSCoordinateSystem": "MNI152NLin2009aSym",
            "NIRSCoordinateUnits": "mm",
        })
    except Exception as e:
        print(f"WARNING {bid}: could not write hardware files ({e})")


# =============================================================================
# 6. MAIN
# =============================================================================


def convert_participant(file_id, snirf_files, args, out_dir):
    """Convert all runs of one participant. Returns the participant ID (sub-XXX)."""
    bid = f"sub-{file_id.replace('P', '')}"
    dest = out_dir / bid / "nirs"
    dest.mkdir(parents=True, exist_ok=True)

    scans = []
    for src in sorted(f for f in snirf_files if f.name.startswith(file_id)):
        run = detect_run(src.name)
        new_name = f"{bid}_task-motor_run-{run}_nirs.snirf"
        shutil.copy(src, dest / new_name)

        # Channels/optodes/coordsystem do not depend on the run, so the
        # (identical) files of the second run simply overwrite those of the first.
        write_hardware_files(args.originals, dest / new_name, file_id, bid, dest)

        acq_time, aurora_version = read_scan_info(src)
        write_json(dest / f"{bid}_task-motor_run-{run}_nirs.json", NIRS_SIDECAR)
        scans.append({
            "filename": f"nirs/{new_name}",
            "acq_time": acq_time,
            "software_version": aurora_version,
        })

    write_events(dest, bid)
    if scans:
        write_tsv(out_dir / bid / f"{bid}_scans.tsv", scans)
    return bid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the raw fNIRS recordings to BIDS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--originals", type=Path, required=True,
                        help="folder with the original exports (probeInfo.mat files)")
    parser.add_argument("--snirf", type=Path, required=True,
                        help="folder with the SNIRF files (5x5 montage)")
    parser.add_argument("--demographics", type=Path, required=True,
                        help="Excel file with age, sex and handedness")
    parser.add_argument("--exclude", type=Path, required=True,
                        help="Excel file with an 'exclude' column")
    parser.add_argument("--output", type=Path, required=True,
                        help="target folder (BIDS root)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output

    excluded_ids = load_excluded_ids(args.exclude)
    demographics, demo_id_col = read_table(args.demographics)

    snirf_files = list(args.snirf.rglob("*.snirf"))
    file_ids = sorted({f.name.split("_")[0] for f in snirf_files})
    file_ids = [i for i in file_ids if digits(i) not in excluded_ids]

    print(f"Participants to convert: {len(file_ids)}")
    if not file_ids:
        print("No matching participants found!")
        return

    # Dataset-level files
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README").write_text(README_CONTENT, encoding="utf-8")
    write_json(out_dir / "dataset_description.json", DATASET_DESCRIPTION)
    write_json(out_dir / "participants.json", PARTICIPANTS_METADATA)

    # Participant-level files
    participants = []
    for file_id in file_ids:
        bid = convert_participant(file_id, snirf_files, args, out_dir)

        match = demographics[demographics[demo_id_col].astype(str).str.contains(file_id.replace("P", ""))]
        if not match.empty:
            row = match.iloc[0].to_dict()
            participants.append({
                "participant_id": bid,
                "age": row.get("age", "n/a"),
                "sex": row.get("sex", "n/a"),
                "handedness": row.get("handedness", "n/a"),
            })

    if participants:
        write_tsv(out_dir / "participants.tsv", participants)

    print("BIDS export finished.")


if __name__ == "__main__":
    main()
