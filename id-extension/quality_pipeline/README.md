# fNIRS signal quality and GLM analysis: Standard-density (SD) vs. Increased-density (ID)

Analysis code for a study comparing two fNIRS optode configurations, Standard-density (SD, 13 channels) and an Increased-density montage (ID, 20 channels), during a motor task (hand clenching, finger tapping). The SD channels are a subset of the ID channels.

Preprocessing, quality assessment and single-subject GLMs were run in [Satori](https://www.brainvoyager.com/satori) (Brain Innovation). The Satori workflows are provided as `.flow` files. Signal quality metrics, figures and group statistics are computed with the Python scripts in this repository.

## Repository contents

| File | Purpose |
|---|---|
| `QualityAssessment_MotionCorrection.flow` | Satori: raw data to optical density, spike removal, TDDR motion correction, trimming. Saves motion-corrected OD data. |
| `SCI_0.75.flow` | Satori: loads motion-corrected OD data and applies SCI channel rejection (threshold 0.75). Exports the SCI text files used below. |
| `SingleSubjectGLM.flow` | Satori: single-subject GLM with two-gamma HRF and three contrasts (1 > Baseline, 2 > Baseline, 1 > 2). |
| `snirf_io.py` | Minimal SNIRF reader (h5py + numpy). |
| `compute_snr_cv.py` | Computes channel-level SNR and CV and pair-level coupling SNR (cSNR), and merges them with the SCI values. |
| `signal_quality_figure.ipynb` | Figure: SCI per channel, SNR and CV for ID and SD. |
| `scatter_sd_vs_id.py` | Figure: decoding accuracy per participant, SD vs. ID. |

## Requirements

Python 3.9 or newer and the packages in `requirements.txt`. `compute_snr_cv.py` needs NumPy 2.0 or newer (`np.trapezoid`). Satori is only needed to reproduce the preprocessing and GLM steps.

```
pip install -r requirements.txt
```

## Workflow

1. **Motion correction** (`QualityAssessment_MotionCorrection.flow`). Raw data, conversion to OD, spike removal, TDDR (with high-frequency restoration), trimming around the task triggers. The motion-corrected OD files are saved.
2. **SCI** (`SCI_0.75.flow`). The motion-corrected OD files are loaded and channels with SCI < 0.75 are rejected. The SCI values per channel pair are exported as `.txt` files (one line per pair, format `CH: 1 SCI: 0.720734`).
3. **SNR, CV and cSNR** (`compute_snr_cv.py`). Uses the *raw* (not motion-corrected) trimmed SNIRF files, because SNR and CV must be computed on unprocessed intensity. SCI is taken from the text files of step 2.

   ```
   python compute_snr_cv.py --snirf-dir <raw_trimmed_snirf> --sci-dir <sci_txt> --out-dir <output>
   ```

   Outputs: `fNIRSQA_output.csv` (per channel and wavelength: SNR, CV) and `all_runs_coupling_metrics.csv` (per source-detector pair: SCI, cSNR, quality labels).
4. **Signal quality figure** (`signal_quality_figure.ipynb`). Select the folder with the two CSV files and an output folder in the dialogs. Writes a PDF and a 600 dpi PNG.
5. **Single-subject GLM** (`SingleSubjectGLM.flow`).
6. **Accuracy scatter plot** (`scatter_sd_vs_id.py`):

   ```
   python scatter_sd_vs_id.py path/to/accuracy_stats.csv
   ```

   Required columns: `acc_sd`, `acc_id`, `paired_p_id_vs_sd_sig_raw`. Optional: `participant`, `diff_id_minus_sd`. The figure is created at its final size (12.5 x 8 cm).

## Metrics

- **SNR (dB)** per channel: `20 * log10(mean / std)` of the raw intensity.
- **CV (%)** per channel: `std / mean * 100`.
- **cSNR (dB)** per source-detector pair: Welch PSD of both wavelengths (`nperseg` = 300); cardiac band 0.8 to 2.0 Hz versus noise bands 0.1 to 0.5 Hz and 2.1 to 2.4 Hz (after Raible et al.); powers are averaged over the two wavelengths.
- **SCI** is read from the Satori export. Quality labels: SCI >= 0.75 strong, >= 0.5 usable, otherwise poor; cSNR >= 13 dB good, >= 2.5 dB usable, otherwise poor.

## Data and file naming

- SNIRF files must follow BIDS-style names containing `sub-XXX` and `run-N` (for example `sub-001_task-motor_run-1_nirs.snirf`). SCI text files are matched to SNIRF files by these two tokens.
- The SCI file entries are matched to channel pairs by position in the sorted list of unique (source, detector) pairs. This assumes that Satori numbers the pairs in the same order.
- Run 1 is labelled Run A and run 2 Run B in the figures.
- In `signal_quality_figure.ipynb` the mapping from SD to ID channel names is defined in `SD_TO_ID`.

## Satori workflows

The `.flow` files contain the absolute paths of the machine they were created on (`Z:/...`). Open them in Satori and update the file lists and output folders before running.

## Citation

Please add the citation of the associated article and the data repository here.

## License

Please add a license before publishing.
