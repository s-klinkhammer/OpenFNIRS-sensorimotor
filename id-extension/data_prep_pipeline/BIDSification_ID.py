"""BIDSification of the fNIRS motor dataset (ID montage, 8x8).

Converts SNIRF recordings into a BIDS dataset: copies the SNIRF files, writes
channels/optodes/events/scans/participants files and the dataset metadata.

The ID and SD versions of this script are identical except for the
"Montage-specific settings" block below.
"""

import json
import os
import re
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import h5py
import pandas as pd
from scipy.io import loadmat

# =================================================================
# MONTAGE-SPECIFIC SETTINGS (the only block that differs between SD and ID)
# =================================================================

MONTAGE_LAYOUT = "8x8"

# Optional start locations for the file dialogs (None = default location)
DEFAULT_ORIGINALS_DIR = None
DEFAULT_SNIRFS_DIR = None
DEFAULT_TARGET_DIR = None
DEFAULT_DEMO_EXCEL = None
DEFAULT_EXCL_EXCEL = None

# Short channels as (source, detector) in the numbering of the SNIRF files.
# ID: the files keep the original 8x8 numbering, so the short channel is S8-D3.
# Add further pairs here if the 8x8 setup has more short channels.
SHORT_CHANNELS = {(8, 3)}

# Only needed for SD, where the SNIRF files are renumbered. None means that the
# SNIRF numbering equals the original numbering used in probeInfo.mat.
ORIGINAL_SOURCES = None
ORIGINAL_DETECTORS = None

GENERATED_BY_DESCRIPTION = (
    "Custom in-house pipeline converting raw fNIRS .snirf recordings and .tri "
    "trigger files into BIDS format, including event extraction from SNIRF "
    "stim data."
)

# =================================================================
# DATASET TEXTS AND METADATA
# =================================================================

readme_content = """# NIRS sensorimotor tasks: Hand clenching & Finger tapping

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

dataset_info = {
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
        "This work was funded through The Netherlands Organization for Scientific "
        "Research (NWO; Vidi-Grant No. VI.Vidi.191.210 to Bettina Sorger)"
    ],
    "GeneratedBy": [
        {
            "Name": "Custom BIDSification script (Python)",
            "Description": GENERATED_BY_DESCRIPTION,
        }
    ],
}

participants_metadata = {
    "participant_id": {"Description": "Unique participant identifier"},
    "age": {"Description": "Age of the participant", "Units": "years"},
    "sex": {"Description": "Biological sex", "Levels": {"f": "female", "m": "male"}},
    "handedness": {"Description": "Handedness", "Levels": {"right": "right", "left": "left"}},
}

# Channel and optode counts are placeholders here and are filled in per file
# from the SNIRF data (see build_run_metadata).
bids_metadata_template = {
    "TaskName": "motor",
    "Manufacturer": "NIRx",
    "ManufacturersModelName": "NIRSport2",
    "SamplingFrequency": 10.1725,
    "NIRSChannelCount": None,
    "NIRSSourceOptodeCount": None,
    "NIRSDetectorOptodeCount": None,
    "ACCELChannelCount": 3,
    "GYROChannelCount": 3,
    "CapManufacturer": "EasyCap",
    "SourceType": "LED",
    "DetectorType": "SiPD",
    "ShortChannelCount": None,
    "NIRSPlacementScheme": "10-5",
    "SoftwareVersions": "Aurora version(s) can be found in individual scan.tsv files",
    "InstitutionName": "Maastricht University",
    "InstitutionalDepartmentName": "Department of Cognitive Neuroscience at the Faculty of Psychology and Neuroscience",
}

TASK_DURATION = 10.0  # seconds

# Maps the stim names stored in the SNIRF files to BIDS trial types
TRIGGER_LABELS = {
    "1": "hand_clenching",
    "2": "finger_tapping",
}

# =================================================================
# FUNCTIONS
# =================================================================


def _decode_stim_name(raw_name):
    """Convert the raw HDF5 'name' field of a stim group into a plain string."""
    if isinstance(raw_name, bytes):
        return raw_name.decode()
    if hasattr(raw_name, "__len__") and not isinstance(raw_name, str):
        first = raw_name[0] if len(raw_name) else raw_name
        return _decode_stim_name(first)
    return str(raw_name)


def process_events(dest, bid):
    """Write events.tsv files based on the stim groups of the copied SNIRF files.

    The stim names in the SNIRF files are normalized ('1' = hand clenching comes
    first, '2' = finger tapping second), whereas the raw trigger codes in the
    .tri files varied between participants. Runs that do not match this pattern
    (exactly two stim groups, '1' before '2') are skipped with a warning.
    """
    for snirf_path in sorted(dest.glob(f"{bid}_task-motor_run-*_nirs.snirf")):
        run = snirf_path.stem.split("run-")[1].split("_")[0]

        stims = []
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
            print(f"WARNING {snirf_path.name}: could not read stim data ({e}), skipped!")
            continue

        if len(stims) != 2:
            print(f"WARNING {snirf_path.name}: {len(stims)} stim groups instead of 2, skipped, check manually!")
            continue

        first_group, second_group = sorted(stims, key=lambda s: s[1].min())
        if first_group[0] != "1" or second_group[0] != "2":
            print(
                f"WARNING {snirf_path.name}: unexpected order/codes "
                f"('{first_group[0]}' before '{second_group[0]}', expected '1' before '2'), "
                f"skipped, check manually!"
            )
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

        pd.DataFrame(events).sort_values("onset").reset_index(drop=True).to_csv(
            dest / f"{bid}_task-motor_run-{run}_events.tsv", sep="\t", index=False
        )


def build_channel_rows(nirs_data):
    """Return the channel table rows (fNIRS channels plus IMU channels)."""
    rows = []
    for key in (k for k in nirs_data.keys() if k.startswith("measurementList")):
        ml = nirs_data[key]
        s = int(ml["sourceIndex"][0])
        d = int(ml["detectorIndex"][0])
        wavelength = 760 if int(ml["wavelengthIndex"][0]) == 1 else 850
        rows.append({
            "name": f"S{s}-D{d}_{wavelength}",
            "type": "NIRSCWAMPLITUDE",
            "source": f"S{s}",
            "detector": f"D{d}",
            "wavelength_nominal": wavelength,
            "units": "V",
            "short_channel": "true" if (s, d) in SHORT_CHANNELS else "false",
            "component": "n/a",
            "_sort": (s, d, wavelength),
        })

    rows.sort(key=lambda r: r["_sort"])
    for row in rows:
        del row["_sort"]

    for imu_type, units in (("ACCEL", "m/s^2"), ("GYRO", "rad/s")):
        for axis in ("x", "y", "z"):
            rows.append({
                "name": f"{imu_type}_{axis.upper()}",
                "type": imu_type,
                "source": "n/a",
                "detector": "n/a",
                "wavelength_nominal": "n/a",
                "units": units,
                "short_channel": "false",
                "component": axis,
            })
    return rows


def build_optode_rows(prefix, optode_type, positions, original_ids, labels):
    """Return optode table rows for all sources or all detectors.

    original_ids maps the SNIRF index to the index in probeInfo.mat (1-based),
    or is None if both numberings are identical.
    """
    rows = []
    for idx in range(positions.shape[0]):
        original_idx = original_ids[idx] if original_ids else idx + 1
        label_idx = original_idx - 1
        rows.append({
            "name": f"{prefix}{idx + 1}",
            "type": optode_type,
            "x": "n/a",
            "y": "n/a",
            "z": "n/a",
            "template_x": round(positions[idx][0], 6),
            "template_y": round(positions[idx][1], 6),
            "template_z": round(positions[idx][2], 6),
            "description": labels[label_idx] if label_idx < len(labels) else "[]",
        })
    return rows


def process_hardware(meta_path, snirf_path, sid_f, bid, dest):
    """Write channels.tsv, optodes.tsv and coordsystem.json for one participant."""
    probe_files = list(meta_path.rglob(f"*{sid_f}*probeInfo.mat"))
    if not probe_files:
        probe_files = list(meta_path.rglob("*probeInfo.mat"))
    if not probe_files:
        return

    try:
        mat = loadmat(probe_files[0], struct_as_record=False, squeeze_me=True)
        p_info = mat["probeInfo"]

        with h5py.File(snirf_path, "r") as f:
            pd.DataFrame(build_channel_rows(f["nirs/data1"])).to_csv(
                dest / f"{bid}_task-motor_channels.tsv", sep="\t", index=False
            )

            optodes = build_optode_rows(
                "S", "source", f["nirs/probe/sourcePos3D"][:],
                ORIGINAL_SOURCES, p_info.probes.labels_s,
            ) + build_optode_rows(
                "D", "detector", f["nirs/probe/detectorPos3D"][:],
                ORIGINAL_DETECTORS, p_info.probes.labels_d,
            )
            pd.DataFrame(optodes).to_csv(dest / f"{bid}_optodes.tsv", sep="\t", index=False)

        with open(dest / f"{bid}_coordsystem.json", "w") as jf:
            json.dump(
                {"NIRSCoordinateSystem": "MNI152NLin2009aSym", "NIRSCoordinateUnits": "mm"},
                jf,
                indent=4,
            )

    except Exception as e:
        print(f"Hardware error {bid}: {e}")


def build_run_metadata(snirf_file):
    """Build the _nirs.json content, with counts read from the SNIRF file."""
    with h5py.File(snirf_file, "r") as fs:
        n_src = fs["nirs/probe/sourcePos3D"].shape[0]
        n_det = fs["nirs/probe/detectorPos3D"].shape[0]
        n_cols = fs["nirs/data1/dataTimeSeries"].shape[1]

    meta = dict(bids_metadata_template)
    meta.update({
        "NIRSChannelCount": n_cols + 6,  # fNIRS columns + 3 ACCEL + 3 GYRO
        "NIRSSourceOptodeCount": n_src,
        "NIRSDetectorOptodeCount": n_det,
        "ShortChannelCount": len(SHORT_CHANNELS) * 2,  # one row per wavelength
    })
    return meta


def read_scan_info(snirf_file):
    """Return (acquisition time, Aurora version) from the SNIRF metadata tags."""
    acq_time = "n/a"
    aurora_version = "n/a"

    def as_str(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    try:
        with h5py.File(snirf_file, "r") as f:
            tags = f.get("nirs/metaDataTags")
            if tags:
                if "MeasurementDate" in tags and "MeasurementTime" in tags:
                    acq_time = f"{as_str(tags['MeasurementDate'][0])}T{as_str(tags['MeasurementTime'][0])}"
                if "AuroraVersion" in tags:
                    parts = as_str(tags["AuroraVersion"][0]).split("-")
                    short_version = "-".join(parts[:2]) if len(parts) > 1 else parts[0]
                    aurora_version = f"Aurora {short_version}"
    except Exception as e:
        print(f"Metadata error {snirf_file.name}: {e}")

    return acq_time, aurora_version


def ask_directory(title, message, initialdir):
    messagebox.showinfo(title, message)
    return filedialog.askdirectory(initialdir=initialdir)


def ask_excel(title, message, default_file):
    messagebox.showinfo(title, message)
    exists = bool(default_file) and os.path.exists(default_file)
    return filedialog.askopenfilename(
        initialdir=os.path.dirname(default_file) if exists else None,
        initialfile=os.path.basename(default_file) if exists else "",
        filetypes=[("Excel", "*.xlsx *.xls")],
    )


def read_excel_with_id_column(path):
    """Read an Excel file with lowercase column names and return it with its ID column."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ["participant_id", "id", "subject"]), df.columns[0])
    return df, id_col


# =================================================================
# MAIN
# =================================================================


def run_pipeline():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    meta_d_str = ask_directory(
        "Step 1", "Select the folder with the ORIGINALS containing the .tri and probeInfo files.",
        DEFAULT_ORIGINALS_DIR,
    )
    if not meta_d_str:
        return
    snirf_d_str = ask_directory(
        "Step 2", f"Select folder with the CLEAN snirfs ({MONTAGE_LAYOUT}).", DEFAULT_SNIRFS_DIR
    )
    if not snirf_d_str:
        return
    dst_d_str = ask_directory("Step 3", "Select TARGET folder.", DEFAULT_TARGET_DIR)
    if not dst_d_str:
        return
    ex_demo_p = ask_excel(
        "Step 4", "Select Excel file with the DEMOGRAPHICS (age, sex, handedness).", DEFAULT_DEMO_EXCEL
    )
    ex_exclude_p = ask_excel(
        "Step 5", "Select excel file with the EXCLUDE participants.", DEFAULT_EXCL_EXCEL
    )
    if not (ex_demo_p and ex_exclude_p):
        return

    meta_d, snirf_d, dst_d = Path(meta_d_str), Path(snirf_d_str), Path(dst_d_str)

    # Participants marked in the 'exclude' column are skipped
    df_ex, ex_id_col = read_excel_with_id_column(ex_exclude_p)
    exclude_ids = []
    if "exclude" in df_ex.columns:
        raw_exclude = df_ex[df_ex["exclude"].notna()][ex_id_col].astype(str).tolist()
        exclude_ids = [re.sub(r"\D", "", x) for x in raw_exclude]

    df_demo, demo_id_col = read_excel_with_id_column(ex_demo_p)

    all_snirfs = list(snirf_d.rglob("*.snirf"))
    unique_ids = sorted({f.name.split("_")[0] for f in all_snirfs})
    selected_ids = [sid for sid in unique_ids if re.sub(r"\D", "", sid) not in exclude_ids]

    total_count = len(selected_ids)
    print(f"\nParticipants after exclusion: {total_count}\n")
    if total_count == 0:
        print("No matching participants found!")
        return

    # Terminal input instead of a Tkinter popup (the popup crashes on macOS)
    try:
        start_input = input("Start index [default 1]: ").strip()
        start_idx = int(start_input) if start_input else 1
        end_input = input(f"End index [default {total_count}]: ").strip()
        end_idx = int(end_input) if end_input else total_count
    except ValueError:
        print("Invalid input, please enter numbers.")
        return

    selected_ids = selected_ids[start_idx - 1:end_idx]
    print(f"\nProcessing {len(selected_ids)} participants (index {start_idx} to {end_idx})...\n")

    dst_d.mkdir(parents=True, exist_ok=True)
    (dst_d / "README").write_text(readme_content, encoding="utf-8")
    with open(dst_d / "dataset_description.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4)
    with open(dst_d / "participants.json", "w", encoding="utf-8") as f:
        json.dump(participants_metadata, f, indent=4)

    participant_rows = []

    for sid_f in selected_ids:
        sid_c = sid_f.replace("P", "")
        bid = f"sub-{sid_c}"

        dest = dst_d / bid / "nirs"
        dest.mkdir(parents=True, exist_ok=True)

        scans_list = []
        subject_files = sorted(f for f in all_snirfs if f.name.startswith(sid_f))

        for sp in subject_files:
            run = "1" if any(x in sp.name.lower() for x in ["run1", "runa", "run_1"]) else "2"
            new_name = f"{bid}_task-motor_run-{run}_nirs.snirf"
            shutil.copy(sp, dest / new_name)

            process_hardware(meta_d, dest / new_name, sid_f, bid, dest)

            with open(dest / f"{bid}_task-motor_run-{run}_nirs.json", "w", encoding="utf-8") as f:
                json.dump(build_run_metadata(dest / new_name), f, indent=4)

            acq_time, aurora_version = read_scan_info(sp)
            scans_list.append({
                "filename": f"nirs/{new_name}",
                "acq_time": acq_time,
                "software_version": aurora_version,
            })

        process_events(dest, bid)

        if scans_list:
            pd.DataFrame(scans_list).to_csv(dst_d / bid / f"{bid}_scans.tsv", sep="\t", index=False)

        match = df_demo[df_demo[demo_id_col].astype(str).str.contains(sid_c)]
        if not match.empty:
            row = match.iloc[0].to_dict()
            participant_rows.append({
                "participant_id": bid,
                "age": row.get("age", "n/a"),
                "sex": row.get("sex", "n/a"),
                "handedness": row.get("handedness", "n/a"),
            })

    if participant_rows:
        pd.DataFrame(participant_rows).to_csv(dst_d / "participants.tsv", sep="\t", index=False)

    messagebox.showinfo("Done", "BIDS export finished!")


if __name__ == "__main__":
    run_pipeline()
