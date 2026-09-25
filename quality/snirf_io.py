"""
snirf_io.py
-----------
Minimal, dependency-light reader for the SNIRF format (SNIRF = HDF5 with a
fixed group layout, see https://github.com/fNIRS/snirf).

We deliberately do NOT depend on the `snirf`/pysnirf2 or `mne`/`mne-nirs`
packages. h5py + numpy is enough to pull out exactly what the QC metrics
need, and it keeps the dependency list short for a shared analysis script.

If your SNIRF files deviate from the layout assumed here (e.g. multiple
`nirs` groups, multiple `data` blocks per file, non-continuous-wave data),
this will raise a clear error rather than silently producing wrong numbers
-- please open the file in HDF5-view or Satori to check the structure if
that happens, and adjust `load_snirf` accordingly.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np


@dataclasses.dataclass
class SnirfData:
    file: Path
    fs: float                     # sampling rate, Hz (derived from the time vector)
    time: np.ndarray               # (n_samples,)
    intensity: np.ndarray          # (n_samples, n_channels) raw light intensity
    source_index: np.ndarray       # (n_channels,) 1-based source index per column
    detector_index: np.ndarray     # (n_channels,) 1-based detector index per column
    wavelength_nm: np.ndarray      # (n_channels,) wavelength in nm per column
    stim_onsets: np.ndarray        # (n_events,) onset times, seconds (may be empty)
    stim_durations: np.ndarray     # (n_events,) durations, seconds (may be empty)

    @property
    def channel_label(self) -> np.ndarray:
        return np.array(
            [f"S{s}-D{d}" for s, d in zip(self.source_index, self.detector_index)]
        )


def _read_scalar(group: h5py.Group, name: str):
    """SNIRF stores many fields as length-1 datasets; unwrap them to a Python scalar."""
    val = group[name][()]
    if isinstance(val, np.ndarray):
        val = val.reshape(-1)[0]
    return val


def load_snirf(path: str | Path) -> SnirfData:
    """Load raw intensity + channel geometry + stim table from one .snirf file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        nirs_keys = [k for k in f.keys() if k.startswith("nirs")]
        if len(nirs_keys) != 1:
            raise ValueError(
                f"{path.name}: expected exactly one top-level 'nirs' group, "
                f"found {nirs_keys}. Multi-run SNIRF files aren't handled here yet."
            )
        nirs = f[nirs_keys[0]]

        data_keys = [k for k in nirs.keys() if k.startswith("data")]
        if len(data_keys) != 1:
            raise ValueError(
                f"{path.name}: expected exactly one 'data' block, found {data_keys}."
            )
        data = nirs[data_keys[0]]

        raw = np.asarray(data["dataTimeSeries"])  # (n_samples, n_channels)
        time = np.asarray(data["time"]).reshape(-1)  # (n_samples,)
        if raw.shape[0] != time.shape[0]:
            # SNIRF allows either orientation depending on writer; transpose if needed
            if raw.shape[1] == time.shape[0]:
                raw = raw.T
            else:
                raise ValueError(
                    f"{path.name}: dataTimeSeries shape {raw.shape} doesn't match "
                    f"time vector length {time.shape[0]}."
                )

        n_channels = raw.shape[1]
        ml_keys = sorted(
            [k for k in data.keys() if k.startswith("measurementList")],
            key=lambda k: int(k.replace("measurementList", "")),
        )
        if len(ml_keys) != n_channels:
            raise ValueError(
                f"{path.name}: found {len(ml_keys)} measurementList entries for "
                f"{n_channels} data columns -- mismatch."
            )

        probe = nirs["probe"]
        wavelengths = np.asarray(probe["wavelengths"]).reshape(-1)

        source_index = np.zeros(n_channels, dtype=int)
        detector_index = np.zeros(n_channels, dtype=int)
        wavelength_nm = np.zeros(n_channels, dtype=float)
        for i, key in enumerate(ml_keys):
            ml = data[key]
            source_index[i] = int(_read_scalar(ml, "sourceIndex"))
            detector_index[i] = int(_read_scalar(ml, "detectorIndex"))
            wl_idx = int(_read_scalar(ml, "wavelengthIndex"))
            wavelength_nm[i] = wavelengths[wl_idx - 1]  # SNIRF indices are 1-based

        # Stimulus / trigger table (optional -- some files have none)
        stim_keys = [k for k in nirs.keys() if k.startswith("stim")]
        onsets, durations = [], []
        for key in stim_keys:
            stim_data = np.asarray(nirs[key]["data"])
            if stim_data.ndim == 1:
                stim_data = stim_data.reshape(1, -1)
            for row in stim_data:
                onsets.append(float(row[0]))
                durations.append(float(row[1]) if len(row) > 1 else 0.0)
        onsets = np.array(onsets)
        durations = np.array(durations)
        order = np.argsort(onsets)
        onsets, durations = onsets[order], durations[order]

        fs = 1.0 / np.median(np.diff(time))

        return SnirfData(
            file=path,
            fs=fs,
            time=time,
            intensity=raw,
            source_index=source_index,
            detector_index=detector_index,
            wavelength_nm=wavelength_nm,
            stim_onsets=onsets,
            stim_durations=durations,
        )


def trim_to_stim_window(d: SnirfData, pre: float = 0.0, post: float = 0.0) -> SnirfData:
    """
    Mirrors the Satori 'Trim Data' node in this workflow: cut to
    [first_trigger - pre, last_trigger_end + post]. With pre=post=0 (the
    values used in qc_respra_neurophotonics.flow) this trims to exactly the
    task period, dropping rest/baseline at the start and end.

    If there are no stim markers in the file, returns the data unchanged.
    """
    if d.stim_onsets.size == 0:
        return d
    start = d.stim_onsets[0] - pre
    end = d.stim_onsets[-1] + d.stim_durations[-1] + post
    mask = (d.time >= start) & (d.time <= end)
    return dataclasses.replace(
        d,
        time=d.time[mask],
        intensity=d.intensity[mask, :],
    )
