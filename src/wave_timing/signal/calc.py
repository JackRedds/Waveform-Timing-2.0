import numpy as np
from wave_timing.signal import operation


def find_nearest(sig: np.ndarray, value):
    idx = np.argmin(np.abs(sig - value))
    return idx


def sample_rate(time: np.ndarray):
    sample_rate = len(time) / time[-1]
    nyquist = sample_rate / 2.0
    dt = time[1] - time[0]

    return sample_rate, nyquist, dt


def peak_to_peak(sig: np.ndarray):
    assert len(sig) != 0
    #sig must be numpy array
    peaks, troughs = operation.find_peaks_and_troughs(sig)
    length = len(peaks)
    ave_peak_to_peak = np.zeros(length)
    for i in range(length):
        trough_less = troughs[troughs < peaks[i]]
        if len(trough_less) > 0:
            peak_less = abs(sig[trough_less[-1]]) + abs(sig[peaks[i]])

        trough_greater = troughs[troughs > peaks[i]]
        if len(trough_greater) > 0:
            peak_greater = abs(sig[trough_greater[0]]) + abs(sig[peaks[i]])

        if len(trough_less) == 0 and len(trough_greater) == 0:
            ave_pk = np.nan

        elif len(trough_less) == 0:
            ave_pk = peak_greater

        elif len(trough_greater) == 0:
            ave_pk = peak_less

        else:
            ave_pk = (peak_less + peak_greater) / 2

        ave_peak_to_peak[i] = ave_pk
    return ave_peak_to_peak, peaks
