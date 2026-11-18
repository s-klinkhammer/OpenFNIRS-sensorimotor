# fNIRS sensorimotor tasks: data preparation scripts

Scripts for preparing and organizing the fNIRS dataset *"NIRS sensorimotor tasks: Hand clenching & Finger tapping"* (NIRx NIRSport2, Maastricht University). The dataset exists in two montages, with one being a subset of the other:

| Montage | Name | Layout | Short channel |
|---------|------|--------|---------------|
| **ID** | Increased Density | 8x8 (full recording) | S8-D3 |
| **SD** | Standard Density | 5x5 (subset of the 8x8 recording) | S5-D2 (= S8-D3 in the original numbering) |

## Pipeline overview

Run the steps in this order, once for ID and once for SD:

| Step | Script / file | Input | Output |
|------|---------------|-------|--------|
| 1 | `reduce_snirf.py` | full 8x8 `.snirf` files | RAW-only `.snirf` files of the chosen montage |
| 2 | `BIDSification_ID.py` or `BIDSification_SD.py` | RAW-only `.snirf` files, `probeInfo.mat`, demographics and exclusion Excel files | BIDS raw dataset |
| 3 | `PreprocessingFlow_Satori.flow` | `.snirf` files | preprocessed `*_PPT.snirf` files |
| 4 | `BIDSification_derivatives.py` | `*_PPT.snirf` files, exclusion Excel file | `derivatives/nirs-preproc/` inside the BIDS dataset |

## Requirements

Python 3 with:

```
pip install h5py numpy pandas scipy openpyxl
```

`tkinter` is part of the standard Python installation (on Linux: `sudo apt install python3-tk`). Step 3 needs Satori 2.2.4 (Brain Innovation).

## Step 1: `reduce_snirf.py`

Creates reduced copies of the SNIRF files: keeps only the `RAW` data (OD and HbO/HbR are removed), removes the optodes that are not part of the chosen montage, and renumbers sources, detectors and channels without gaps. The original files are never modified. The input must be the full 8x8 recording in its original numbering.

```
python reduce_snirf.py --input <input folder> --output <output folder> --setup ID_8x8
python reduce_snirf.py --input <input folder> --output <output folder> --setup SD_5x5
```

- `ID_8x8`: nothing is removed.
- `SD_5x5`: sources S2, S4, S5 and detectors D2, D4, D7 are removed.

Input and output folder must differ. Run `python reduce_snirf.py --help` for all options.

## Step 2: `BIDSification_ID.py` / `BIDSification_SD.py`

Converts the RAW-only SNIRF files into a BIDS dataset. The two scripts are identical except for the block *"Montage-specific settings"* at the top (layout, short channel, optode renumbering for SD). Use the script that matches the montage from step 1.

```
python BIDSification_ID.py
python BIDSification_SD.py
```

The script asks for the following in file dialogs:

1. Folder with the originals (containing the `probeInfo.mat` files, used for the optode labels)
2. Folder with the RAW-only SNIRF files from step 1
3. Target folder (the BIDS root)
4. Excel file with demographics
5. Excel file with the participants to exclude

Then it asks in the terminal for a start and end index (press Enter to process all participants).

**Written per participant** (`sub-XXX/nirs/`): `*_nirs.snirf`, `*_nirs.json`, `*_channels.tsv`, `*_optodes.tsv`, `*_coordsystem.json`, `*_events.tsv`, plus `sub-XXX_scans.tsv`. **Written once:** `README`, `dataset_description.json`, `participants.json`, `participants.tsv`.

Input file names must start with the participant ID (`P001_...`) and contain the run as `RunA`/`Run1`/`Run_1` for run 1. Everything else is assigned to run 2.

Events are read from the stim groups in the SNIRF files (`1` = hand clenching, `2` = finger tapping, 10 s each). A run is skipped with a warning if it does not have exactly two stim groups with `1` before `2`. Check the terminal output for warnings.

## Step 3: `PreprocessingFlow_Satori.flow`

Satori preprocessing flow. Open it in Satori 2.2.4. Load your own SNIRF files before running. The full pipeline (OD, mBLL, spike removal, TDDR, short-channel regression, filtering, z-transformation, trimming, epoching) is described in `derivatives/nirs-preproc/README`, which is written by step 4.

## Step 4: `BIDSification_derivatives.py`

Organizes the preprocessed files into the BIDS derivatives structure. Works for both montages: select the matching preprocessed folder and the matching BIDS root.

```
python BIDSification_derivatives.py
```

Asks in file dialogs for (1) the folder with the preprocessed files, (2) the BIDS root of the same montage, (3) the exclusion Excel file. Input file names must look like `sub-001_task-motor_run-1_nirs_PPT.snirf`; other names are listed in the terminal and not copied. Output goes to `<BIDS root>/derivatives/nirs-preproc/`.

## Exclusion list (`exclude.xlsx`)

Used in steps 2 and 4. Excel file with a column named `exclude` that contains the participant IDs to leave out (e.g. `P010`). Only the digits are compared, so `P010` matches `sub-010`. If the file has an additional ID column (`participant_id`, `id` or `subject`), every participant with a non-empty `exclude` cell is excluded.

## Demographics file

Used in step 2. Excel file with an ID column (`participant_id`, `id` or `subject`; otherwise the first column is used) and the columns `age`, `sex` (`f`/`m`) and `handedness` (`right`/`left`).

## Channel mapping between SD and ID

The SD dataset was created by computationally removing optodes from ID, and Satori
renumbers channels automatically when optodes are removed. Channel numbers therefore
differ between SD and ID (e.g. SD CH 4 = S2-D3 corresponds to ID CH 6 = S3-D5).

`channel_mapping.tsv` lists, for each of the 13 SD channels, the corresponding ID channel
(matched via source-detector pair). Channel-wise results of shared channels are identical
in both datasets, since all preprocessing steps operate channel-wise and the same
short-separation channel was used in both configurations.

## Citation and contact

Citation: *[to be added]*. Contact: Bettina Sorger, Maastricht University (b.sorger@maastrichtuniversity.nl).
