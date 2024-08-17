import numpy as np
from scipy import signal, fft
from wave_timing.math import analysis
from wave_timing.signal import calc


class FIRBandPass(object):

    def __init__(self, taps, cutoff, fs):
        """
        A FIR band pass filter.

        :param taps: The length of the filter.
        :param cutoff: Cutoff frequency [low_cut, high_cut]. (Hz)
        :param fs: The signal sample rate (Sa/s)
        """
        self.taps = taps
        self.cutoff = cutoff
        self.fs = fs

        # Make the FIR filter.
        self.nyq = fs / 2.0
        self.taps = signal.firwin(numtaps=taps,
                           cutoff=cutoff,
                           fs=fs,
                           pass_zero=False)


    def filter(self, v, trim=False):
        """
        Filter the given signal.

        :param v: The signal to filter.
        :param trim: If True, use the value nan at the beginning and ending of
            the filtered signal where boundary effects are visible.
        :return: The filtered signal.
        """
        v_filtered = np.convolve(self.taps, v, mode='same')
        if trim:
            v_filtered[:self.taps // 2] = np.nan
            v_filtered[-self.taps // 2:] = np.nan
        return v_filtered


    def plot_gain(self):
        """
        Plot the frequency response of the filter.

        :return: A tuple (fig, ax) where fig is the figure and ax is the axis
            the plot is on.
        """
        w, h = signal.freqz(self.taps, 1.0)

        fig, ax = plt.subplots()
        ax.plot(w / max(w) * self.nyq, abs(h))
        ax.set_ylabel('Gain')
        ax.set_xlabel(r'Freq (Hz)')
        ax.set_title(r'Frequency response')

        return fig, ax


def sliding_fft(sig: np.ndarray, time: np.ndarray, window_size=5000, hop_size=2500, norm=True):
    assert len(sig) != 0
    assert len(time) != 0
    assert window_size > 0
    assert hop_size > 0

    # sig and time must be numpy arrays
    # possibly work on normalization
    #sig = wave_normalization(sig)

    signal_length = len(sig)
    samp_rt = calc.sample_rate(time)[0]
    num_windows = (signal_length - window_size) // hop_size + 1
    xf = np.abs(fft.fftfreq(window_size, 1.0 / samp_rt)[:window_size//2])
    time_spec = np.linspace(time[0], time[-1], num_windows)
    spectra = []
    time_range = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        if norm:
            sig_cut = analysis.wave_normalization(sig[start:end])
        else:
            sig_cut = sig[start:end]
        window = sig_cut * np.hanning(window_size)
        spectrum = fft.fft(window) / window_size
        spectrum = np.abs(spectrum[:window_size//2])

        spectra.append(spectrum)
        time_range.append([start, end])

    spectra = np.array(spectra)
    time_range = np.array(time_range)

    return spectra, time_spec, xf, time_range
