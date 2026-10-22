"""Preprocessing methods for one-dimensional ABP waveforms."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, medfilt, savgol_filter
import pywt


def _validate_signal(signal):
    x = np.asarray(signal, dtype=float).flatten()
    if x.size < 3:
        raise ValueError("Signal must contain at least 3 samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Signal contains NaN or infinite values. Handle missing segments "
            "explicitly before filtering; do not silently join separated segments."
        )
    return x


def moving_average(signal, window=5):
    x = _validate_signal(signal)
    if window < 1:
        raise ValueError("window must be >= 1")
    return np.convolve(x, np.ones(window, dtype=float) / window, mode="same")


def median_filter(signal, window=5):
    x = _validate_signal(signal)
    if window < 1 or window % 2 == 0:
        raise ValueError("Median window must be a positive odd integer.")
    return medfilt(x, kernel_size=window)


def butterworth_lowpass(signal, fs, cutoff=10.0, order=4):
    x = _validate_signal(signal)
    if fs <= 0 or not 0 < cutoff < fs / 2:
        raise ValueError("Require fs > 0 and 0 < cutoff < Nyquist frequency.")
    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def savitzky_golay(signal, window=7, polyorder=2):
    x = _validate_signal(signal)
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and greater than polyorder.")
    if window > x.size:
        raise ValueError("window cannot exceed signal length.")
    return savgol_filter(x, window_length=window, polyorder=polyorder)


def wavelet_denoise(signal, wavelet="db4", level=3):
    x = _validate_signal(signal)
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(x.size, wavelet_obj.dec_len)
    use_level = min(level, max_level)
    if use_level < 1:
        return x.copy()
    coeffs = pywt.wavedec(x, wavelet_obj, level=use_level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(x.size))
    denoised = [coeffs[0]]
    denoised.extend(pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:])
    return pywt.waverec(denoised, wavelet_obj)[:x.size]


def hampel_filter(signal, window=5, n_sigma=3.0):
    """Replace local outliers using a centered odd window."""
    x = _validate_signal(signal)
    if window < 3 or window % 2 == 0:
        raise ValueError("Hampel window must be an odd integer >= 3.")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    result = x.copy()
    half = window // 2
    for i in range(half, len(x) - half):
        local = x[i-half:i+half+1]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0 and abs(x[i] - med) > n_sigma * robust_sigma:
            result[i] = med
    return result


def preprocess_all(signal, fs):
    """Return all six preprocessing outputs in a named dictionary."""
    x = _validate_signal(signal)
    return {
        "Raw": x.copy(),
        "Moving Average": moving_average(x, window=5),
        "Median Filter": median_filter(x, window=5),
        "Butterworth": butterworth_lowpass(x, fs=fs, cutoff=10.0, order=4),
        "Savitzky-Golay": savitzky_golay(x, window=7, polyorder=2),
        "Wavelet": wavelet_denoise(x, wavelet="db4", level=3),
        "Hampel": hampel_filter(x, window=5, n_sigma=3.0),
    }
