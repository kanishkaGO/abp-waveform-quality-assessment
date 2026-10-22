"""Exploratory signal-quality features; not a clinically validated SQI."""
import numpy as np
from scipy.signal import find_peaks


def amplitude_outlier_fraction(signal, window=101, n_sigma=3.0):
    """Return fraction of samples flagged against local median/MAD."""
    x = np.asarray(signal, dtype=float).flatten()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Signal must be non-empty and finite.")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")
    half = window // 2
    flagged = np.zeros(x.size, dtype=bool)
    for i in range(half, len(x) - half):
        local = x[i-half:i+half+1]
        med = np.median(local)
        sigma = 1.4826 * np.median(np.abs(local - med))
        if sigma > 0 and abs(x[i] - med) > n_sigma * sigma:
            flagged[i] = True
    return float(np.mean(flagged))


def candidate_peak_summary(signal, fs, min_distance_s=0.35, prominence_mmhg=10.0):
    """Detect candidate systolic peaks; parameters require record-specific validation."""
    x = np.asarray(signal, dtype=float).flatten()
    if x.size < 3 or fs <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("Provide >=3 finite samples and a positive sampling rate.")
    peaks, properties = find_peaks(
        x,
        distance=max(1, int(min_distance_s * fs)),
        prominence=prominence_mmhg,
    )
    intervals = np.diff(peaks) / fs
    return {
        "candidate_peak_count": int(len(peaks)),
        "mean_interval_s": float(np.mean(intervals)) if intervals.size else np.nan,
        "mean_rate_bpm": (
            float(60 / np.mean(intervals))
            if intervals.size and np.mean(intervals) > 0 else np.nan
        ),
        "peak_indices": peaks,
        "peak_prominences_mmhg": properties.get("prominences", np.array([])),
    }
