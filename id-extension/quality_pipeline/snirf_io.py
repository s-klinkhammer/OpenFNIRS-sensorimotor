"""Minimal SNIRF reader (h5py + numpy only).

Reads raw continuous-wave intensity, channel geometry and the stimulus table
from a SNIRF file (https://github.com/fNIRS/snirf). Files with more than one
`nirs` group or more than one `data` block are rejected with an error.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np


@dataclasses.dataclass
class SnirfData:
    file: Path
    fs: float                     # sampling rate in Hz, derived from the time vector
    time: np.ndarray              # (n_samples,)
    intensity: np.ndarray         # (n_samples, n_channels) raw light intensity
    source_index: np.ndarray      # (n_channels,) 1-based source index per column
    detector_index: np.ndarray    # (n_channels,) 1-based detector index per column
    wavelength_nm: np.ndarray     # (n_channels,) wavelength in nm per column
    stim_onsets: np.ndarray       # (n_events,) onset times in seconds (may be empty)
    stim_durations: np.ndarray    # (n_events,) durations in seconds (may be empty)

    @property
    def channel_label(self) -> np.ndarray:
        return np.array(
            [f"S{s}-D{d}" for s, d in zip(self.source_index, self.detector_index)]
        )


def _read_scalar(group: h5py.Group, name: str):
    """Unwrap a length-1 SNIRF dataset to a scalar."""
    val = group[name][()]
    if isinstance(val, np.ndarray):
        val = val.reshape(-1)[0]
    return val


def load_snirf(path: str | Path) -> SnirfData:
    """Load raw intensity, channel geometry and stimulus table from one .snirf file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        nirs_keys = [k for k in f.keys() if k.startswith("nirs")]
        if len(nirs_keys) != 1:
            raise ValueError(
                f"{path.name}: expected exactly one 'nirs' group, found {nirs_keys}."
            )
        nirs = f[nirs_keys[0]]

        data_keys = [k for k in nirs.keys() if k.startswith("data")]
        if len(data_keys) != 1:
            raise ValueError(
                f"{path.name}: expected exactly one 'data' block, found {data_keys}."
            )
        data = nirs[data_keys[0]]

        raw = np.asarray(data["dataTimeSeries"])
        time = np.asarray(data["time"]).reshape(-1)
        if raw.shape[0] != time.shape[0]:
            if raw.shape[1] == time.shape[0]:
                raw = raw.T
            else:
                raise ValueError(
                    f"{path.name}: dataTimeSeries shape {raw.shape} does not match "
                    f"time vector length {time.shape[0]}."
                )

        n_channels = raw.shape[1]
        ml_keys = sorted(
            [k for k in data.keys() if k.startswith("measurementList")],
            key=lambda k: int(k.replace("measurementList", "")),
        )
        if len(ml_keys) != n_channels:
            raise ValueError(
                f"{path.name}: {len(ml_keys)} measurementList entries for "
                f"{n_channels} data columns."
            )

        wavelengths = np.asarray(nirs["probe"]["wavelengths"]).reshape(-1)

        source_index = np.zeros(n_channels, dtype=int)
        detector_index = np.zeros(n_channels, dtype=int)
        wavelength_nm = np.zeros(n_channels, dtype=float)
        for i, key in enumerate(ml_keys):
            ml = data[key]
            source_index[i] = int(_read_scalar(ml, "sourceIndex"))
            detector_index[i] = int(_read_scalar(ml, "detectorIndex"))
            wl_idx = int(_read_scalar(ml, "wavelengthIndex"))
            wavelength_nm[i] = wavelengths[wl_idx - 1]  # SNIRF indices are 1-based

        onsets, durations = [], []
        for key in (k for k in nirs.keys() if k.startswith("stim")):
            stim_data = np.asarray(nirs[key]["data"])
            if stim_data.ndim == 1:
                stim_data = stim_data.reshape(1, -1)
            for row in stim_data:
                onsets.append(float(row[0]))
                durations.append(float(row[1]) if len(row) > 1 else 0.0)
        onsets = np.array(onsets)
        durations = np.array(durations)
        order = np.argsort(onsets)

        return SnirfData(
            file=path,
            fs=1.0 / np.median(np.diff(time)),
            time=time,
            intensity=raw,
            source_index=source_index,
            detector_index=detector_index,
            wavelength_nm=wavelength_nm,
            stim_onsets=onsets[order],
            stim_durations=durations[order],
        )
