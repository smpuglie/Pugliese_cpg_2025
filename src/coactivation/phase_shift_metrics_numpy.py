import numpy as np
import scipy


def phase_shift_metric_CrossCorr(x1, x2, dt):
    """Compute phase shift metric based on cross-correlation.

    Args:
        x1 (array): First time series data.
        x2 (array): Second time series data.
        dt (float): Time step size.

    Returns:
        float: Phase shift metric value.
    """
    n = len(x1)
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    f1 = np.fft.fft(x1)
    f2 = np.fft.fft(x2)
    freqs = np.fft.fftfreq(n, d=dt)
    target_freq1 = np.abs(freqs[np.argmax(np.abs(f1))])
    target_freq2 = np.abs(freqs[np.argmax(np.abs(f2))])
    target_freq = (target_freq1 + target_freq2) / 2
    print(f"target_freq1: {target_freq1}, target_freq2: {target_freq2}, target_freq: {target_freq}")
    period = 1 / target_freq

    corr = scipy.signal.correlate(x1, x2, mode="full")
    # corr = np.fft.ifft(np.conjugate(f1) * f2)
    lags = scipy.signal.correlation_lags(len(x1), len(x2), mode="full")
    corr /= np.std(x1) * np.std(x2) * n  # (n-np.abs(lags)))
    lag = lags[np.argmax(corr)]
    tau = lag * dt
    phase_shift = 2 * np.pi * (tau / period)
    return phase_shift


def phase_shift_metric_FFT(x1, x2, dt):
    """Compute phase shift metric based on FFT.

    Args:
        x1 (array): First time series data.
        x2 (array): Second time series data.
        dt (float): Time step size.

    Returns:
        float: Phase shift metric value.
    """
    n = len(x1)
    f1 = np.fft.fft(x1 - np.mean(x1))
    f2 = np.fft.fft(x2 - np.mean(x2))
    freqs = np.fft.fftfreq(n, d=dt)

    target_freq1 = freqs[np.argmax(np.abs(f1))]
    target_freq2 = freqs[np.argmax(np.abs(f2))]
    target_freq = (target_freq1 + target_freq2) / 2

    idx = (np.abs(freqs - target_freq)).argmin()

    phase_shift = np.angle(f1[idx] / f2[idx])
    return phase_shift


def phase_shift_metric_Hilbert(x1, x2, dt):
    """Compute phase shift metric based on Hilbert transform.

    Args:
        x1 (array): First time series data.
        x2 (array): Second time series data.
        dt (float): Time step size.

    Returns:
        float: Phase shift metric value.
    """
    analytic_signal1 = scipy.signal.hilbert(x1 - np.mean(x1))
    analytic_signal2 = scipy.signal.hilbert(x2 - np.mean(x2))
    phase_shift = np.angle(analytic_signal1 / analytic_signal2)
    phase_shift = np.mean(phase_shift)
    return phase_shift


def phase_shift_metric_Hilbert_from_FFT(x1, x2, dt):
    """Compute phase shift metric based on Hilbert transform.

    Args:
        x1 (array): First time series data.
        x2 (array): Second time series data.
        dt (float): Time step size.

    Returns:
        float: Phase shift metric value.
    """
    N1 = len(x1)
    N2 = len(x2)
    f1 = np.fft.fft(x1 - np.mean(x1))
    f2 = np.fft.fft(x2 - np.mean(x2))
    f1[N1 // 2 + 1 :] = 0
    f2[N2 // 2 + 1 :] = 0
    f1[1 : N1 // 2] *= 2
    f2[1 : N2 // 2] *= 2
    analytic_signal1 = np.fft.ifft(f1)
    analytic_signal2 = np.fft.ifft(f2)
    phase_shift = np.angle(analytic_signal1 / analytic_signal2)
    phase_shift = np.mean(phase_shift)
    return phase_shift
