"""Rule-based screening for candidate ABP anomalies; flags require review."""
import numpy as np
import pandas as pd


def abrupt_changes(signal, fs, threshold_mmhg=20.0):
    x = np.asarray(signal, dtype=float).flatten()
    if x.size < 2 or fs <= 0:
        raise ValueError("Need at least 2 samples and fs > 0.")
    finite_pair = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    change = np.abs(np.diff(x))
    idx = np.where(finite_pair & (change > threshold_mmhg))[0]
    return pd.DataFrame({
        "Sample_Index": idx + 1,
        "Time_s": (idx + 1) / fs,
        "Previous_ABP_mmHg": x[idx],
        "Current_ABP_mmHg": x[idx + 1],
        "Absolute_Change_mmHg": change[idx],
    })


def repeated_value_runs(signal, fs, min_samples=5):
    x = np.asarray(signal, dtype=float).flatten()
    if x.size == 0 or fs <= 0:
        raise ValueError("Need a non-empty signal and fs > 0.")
    runs = []
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            length = i - start
            if length >= min_samples:
                runs.append({
                    "Start_Sample": start,
                    "End_Sample": i - 1,
                    "Start_Time_s": start / fs,
                    "End_Time_s": (i - 1) / fs,
                    "Run_Length": length,
                    "Duration_s": (length - 1) / fs,
                    "Value_mmHg": x[start],
                })
            start = i
    return pd.DataFrame(runs)


def basic_signal_checks(signal):
    x = np.asarray(signal, dtype=float).flatten()
    finite = np.isfinite(x)
    finite_values = x[finite]
    return {
        "n_samples": int(x.size),
        "non_finite_samples": int((~finite).sum()),
        "repeated_adjacent_pairs": int(np.sum(x[1:] == x[:-1])),
        "near_zero_samples": int(np.sum(finite & (np.abs(x) < 1e-8))),
        "finite_min": float(np.min(finite_values)) if finite_values.size else np.nan,
        "finite_max": float(np.max(finite_values)) if finite_values.size else np.nan,
    }
