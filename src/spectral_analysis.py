"""Welch PSD and spectral feature extraction."""
import numpy as np
from scipy.signal import welch


def spectral_metrics(signal, fs):
    x = np.asarray(signal, dtype=float).flatten()
    if x.size < 2 or fs <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("Provide >=2 finite samples and a positive sampling rate.")
    x = x - np.mean(x)
    frequencies, psd = welch(x, fs=fs, nperseg=min(2048, len(x)))
    if len(psd) < 2 or not np.all(np.isfinite(psd)):
        raise ValueError("Unable to estimate a valid PSD.")
    df = frequencies[1] - frequencies[0]
    total_sum = np.sum(psd)
    if total_sum <= 0:
        return {
            "Dominant Frequency (Hz)": np.nan,
            "Spectral Entropy": np.nan,
            "Spectral Centroid (Hz)": np.nan,
            "Bandwidth (Hz)": np.nan,
            "Spectral Flatness": np.nan,
            "Band Power 0.5-5 Hz": 0.0,
            "Band Power 5-15 Hz": 0.0,
        }
    p = psd / total_sum
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    entropy_normalized = entropy / np.log2(len(p)) if len(p) > 1 else np.nan
    centroid = np.sum(frequencies * psd) / total_sum
    bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * psd) / total_sum)
    flatness = np.exp(np.mean(np.log(psd + 1e-20))) / (np.mean(psd) + 1e-20)
    positive = frequencies > 0
    dominant = frequencies[positive][np.argmax(psd[positive])] if np.any(positive) else np.nan
    band1 = (frequencies >= 0.5) & (frequencies < 5)
    band2 = (frequencies >= 5) & (frequencies <= 15)
    return {
        "Dominant Frequency (Hz)": float(dominant),
        "Spectral Entropy": float(entropy_normalized),
        "Spectral Centroid (Hz)": float(centroid),
        "Bandwidth (Hz)": float(bandwidth),
        "Spectral Flatness": float(flatness),
        "Band Power 0.5-5 Hz": float(np.sum(psd[band1]) * df),
        "Band Power 5-15 Hz": float(np.sum(psd[band2]) * df),
    }
