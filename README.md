# ABP Waveform Quality Assessment and Artifact Removal

A Python research/learning project for exploratory arterial blood pressure (ABP) waveform quality assessment. It includes signal checks, candidate anomaly screening, multiple preprocessing methods, Welch spectral metrics, and exploratory peak summaries.

> **Important:** This project is not a medical device and does not provide clinical diagnoses. Heuristic flags and SQI-like features are not clinically validated. Changes from the raw signal do not establish denoising accuracy without a suitable clean reference or independent annotations.

## Workflow

1. Load a one-dimensional ABP waveform and its sampling frequency.
2. Inspect statistics and plot the waveform.
3. Flag candidate abrupt changes and repeated-value runs for manual review.
4. Compare moving-average, median, Butterworth, Savitzky–Golay, wavelet, and Hampel methods.
5. Calculate spectral features using Welch's method.
6. Summarize exploratory candidate peak intervals.
7. Validate findings against suitable annotations before making performance claims.

## Repository structure

```text
abp-waveform-quality-assessment/
├── notebooks/ABP_Waveform_Analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── artifact_detection.py
│   ├── spectral_analysis.py
│   └── signal_quality.py
├── results/README.md
├── docs/README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate it on Windows with `.venv\\Scripts\\Activate.ps1`, or on macOS/Linux with `source .venv/bin/activate`. Then install dependencies:

```bash
pip install -r requirements.txt
```

## Input data

The notebook expects a CSV containing a numeric `ABP` column. Set `INPUT_CSV`, `ABP_COLUMN`, and `FS` in the notebook setup cell. The example sampling rate is only a placeholder; confirm the actual value and units using the record metadata. For WFDB records, use `wfdb.rdrecord(...)` and select the ABP channel from the record header.

Keep restricted or large raw clinical files out of this repository. The local `data/` directory and common waveform file types are ignored by Git. Follow the source dataset's license and access rules.

## Methods

- Moving-average and median filtering
- Butterworth low-pass filtering
- Savitzky–Golay smoothing
- Wavelet denoising
- Hampel outlier replacement
- Abrupt-change and repeated-value screening
- Welch PSD, spectral entropy, centroid, bandwidth, flatness, and band power
- Exploratory amplitude outlier fraction and candidate peak-interval summaries

## Interpretation and limitations

- Systolic peaks and diastolic troughs are physiological features, not automatically artifacts.
- Large sample-to-sample changes and repeated values are candidate anomalies, not confirmed artifact labels.
- Filters can remove useful waveform detail or alter peak morphology and timing.
- RMSE, MAE, correlation, and PRD measured against raw data describe change from raw, not accuracy against a clean reference.
- Do not claim sensitivity, specificity, precision, F1, SNR improvement, or timing error without ground truth and a defined evaluation protocol.

## Author

Kanishka Goswami  
M.Tech. Data Science and Artificial Intelligence  
IIIT Naya Raipur
