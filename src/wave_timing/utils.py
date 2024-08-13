import numpy as np
from numpy._typing import _256Bit
import pandas as pd
from pytplot import get_data as get_psp_data
import pyspedas
from scipy import signal
from scipy.fft import fft, fftfreq
import wave_timing.calc as wtc
import matplotlib.pyplot as plt


class get_data:

    def __init__(self, start_time: str, end_time: str, data_file_path='./../data'):
        self.start_time = start_time
        self.end_time = end_time
        self.data_file_path = data_file_path


    def get_vac_data(self):
        data_type = 'dfb_dbm_vac'

        vac_data = pyspedas.psp.fields(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l2',
            time_clip=True,
            file_path=self.data_file_path
            )

        vac1 = get_psp_data('psp_fld_l2_dfb_dbm_vac1').y
        vac2 = get_psp_data('psp_fld_l2_dfb_dbm_vac2').y
        vac3 = get_psp_data('psp_fld_l2_dfb_dbm_vac3').y
        vac4 = get_psp_data('psp_fld_l2_dfb_dbm_vac4').y
        vac5 = get_psp_data('psp_fld_l2_dfb_dbm_vac5').y
        time_TT2000 = get_psp_data('psp_fld_l2_dfb_dbm_vac_time_series_TT2000').times
        time = get_psp_data('psp_fld_l2_dfb_dbm_vac1').v[0]
        start_date = pd.to_datetime(time_TT2000, unit='s')

        dv1 = vac1 - (vac3 + vac4) / 2
        dv2 = (vac3 + vac4) / 2 - vac2
        dv3 = (vac1 + vac2) / 2 - vac3
        dv4 = vac4 - (vac1 + vac2) / 2
        dv5 = (vac1 + vac2 + vac3 + vac4) / 4 - vac5

        dv1 = pd.DataFrame(dv1.T, columns=start_date, index=time)
        dv2 = pd.DataFrame(dv2.T, columns=start_date, index=time)
        dv3 = pd.DataFrame(dv3.T, columns=start_date, index=time)
        dv4 = pd.DataFrame(dv4.T, columns=start_date, index=time)
        dv5 = pd.DataFrame(dv5.T, columns=start_date, index=time)

        return dv1, dv2, dv3, dv4, dv5


    def get_mag_data(self):
        data_type = 'mag_SC_4_Sa_per_Cyc'

        data = pyspedas.psp.fields(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l2',
            time_clip=True,
            file_path=self.data_file_path
            )

        mag_data = get_psp_data('psp_fld_l2_mag_SC_4_Sa_per_Cyc')
        date = pd.to_datetime(mag_data.times, unit='s')
        mag_comp = pd.DataFrame(mag_data.y, columns=['Bx', 'By', 'Bz'], index=date)
        B = np.sqrt(mag_comp.Bx ** 2 + mag_comp.By ** 2 + mag_comp.Bz ** 2)

        mag_comp.insert(3, '|B|', B, True)

        return mag_comp


    def get_sw_data(self):
        data_type = 'sf00_l3_mom'

        data = pyspedas.psp.spi(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l3',
            time_clip=True,
            file_path=self.data_file_path
        )

        sw_data = get_psp_data('psp_spi_VEL_RTN_SUN')
        date = pd.to_datetime(sw_data.times, unit='s')
        Vx = sw_data.y[:, 1]
        Vy = -sw_data.y[:, 2]
        Vz = -sw_data.y[:, 0]
        V = np.array([Vx, Vy, Vz]).T
        sw_comp = pd.DataFrame(V, columns=['Vx', 'Vy', 'Vz'], index=date)
        V = np.sqrt(sw_comp.Vx ** 2 + sw_comp.Vy ** 2 + sw_comp.Vz ** 2)

        sw_comp.insert(3, '|V|', V, True)

        return sw_comp


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


def wave_normalization(wave: np.ndarray):
    mean = np.mean(wave)
    std = np.std(wave)
    return (wave - mean) / std


def find_peaks_and_troughs(sig: np.ndarray):
    assert len(sig) != 0

    dict = {'distance': 50, 'prominence': 0.5 * np.mean(sig)}
    peaks = signal.find_peaks(sig, **dict)[0]
    troughs = signal.find_peaks(-sig, **dict)[0]

    return peaks, troughs


def sliding_fft(sig: np.ndarray, time: np.ndarray, window_size=5000, hop_size=2500, norm=True):
    assert len(sig) != 0
    assert len(time) != 0
    assert window_size > 0
    assert hop_size > 0

    # sig and time must be numpy arrays
    # possibly work on normalization
    #sig = wave_normalization(sig)

    signal_length = len(sig)
    samp_rt = wtc.sample_rate(time)[0]
    num_windows = (signal_length - window_size) // hop_size + 1
    xf = np.abs(fftfreq(window_size, 1.0 / samp_rt)[:window_size//2])
    time_spec = np.linspace(time[0], time[-1], num_windows)
    spectra = []
    time_range = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        if norm:
            sig_cut = wave_normalization(sig[start:end])
        else:
            sig_cut = sig[start:end]
        window = sig_cut * np.hanning(window_size)
        spectrum = fft(window) / window_size
        spectrum = np.abs(spectrum[:window_size//2])

        spectra.append(spectrum)
        time_range.append([start, end])

    spectra = np.array(spectra)
    time_range = np.array(time_range)

    return spectra, time_spec, xf, time_range


def peak_to_peak(sig: np.ndarray):
    assert len(sig) != 0
    #sig must be numpy array
    peaks, troughs = find_peaks_and_troughs(sig)
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


def divisible(denom, divid):
    remainder = divid % denom
    mask = remainder < 10
    if np.all(mask):
        return True
    else:
        return False



def frequency_filter(
        wave1: np.ndarray,
        wave2: np.ndarray,
        wave3: np.ndarray,
        wave4: np.ndarray,
        time: np.ndarray,
        window_size=5000,
        hop_size=2500,
        large_peak_prom=0.25,
        small_peakg_prom=0.025,
        small_peakl_prom=0.1
):

    assert len(wave1) != 0
    assert len(wave2) != 0
    assert len(wave3) != 0
    assert len(wave4) != 0
    assert len(time) != 0

    spectra1, __, xf, time_range = sliding_fft(wave1, time, window_size=window_size, hop_size=hop_size)
    time_range = np.array(time_range)
    spectra2 = sliding_fft(wave2, time, window_size=window_size, hop_size=hop_size)[0]
    spectra3 = sliding_fft(wave3, time, window_size=window_size, hop_size=hop_size)[0]
    spectra4 = sliding_fft(wave4, time, window_size=window_size, hop_size=hop_size)[0]
    single_freq = []

    for t_range, spec1, spec2, spec3, spec4 in zip(time_range, spectra1, spectra2, spectra3, spectra4):
        large_peaks = {'prominence': large_peak_prom}
        small_peaks = {'prominence': [small_peakg_prom, large_peak_prom]}

        large_peaks1, __ = signal.find_peaks(spec1, **large_peaks)
        large_peaks2, __ = signal.find_peaks(spec2, **large_peaks)
        large_peaks3, __ = signal.find_peaks(spec3, **large_peaks)
        large_peaks4, __ = signal.find_peaks(spec4, **large_peaks)

        small_peaks1, __ = signal.find_peaks(spec1, **small_peaks)
        small_peaks2, __ = signal.find_peaks(spec2, **small_peaks)
        small_peaks3, __ = signal.find_peaks(spec3, **small_peaks)
        small_peaks4, __ = signal.find_peaks(spec4, **small_peaks)

        large_peak1_num = len(large_peaks1) == 1
        large_peak2_num = len(large_peaks2) == 1
        large_peak3_num = len(large_peaks3) == 1
        large_peak4_num = len(large_peaks4) == 1

        start, end = t_range

       #fig, ax = plt.subplots(2, 2, sharey='row', figsize=(20, 10))
       #ax[0, 0].plot(time[start:end], wave1[start:end], 'r')
       #ax[0, 0].plot(time[start:end], wave2[start:end], 'k')
       #ax[1, 0].plot(xf, spec1, 'r')
       #ax[1, 0].plot(xf, spec2, 'k')
       #ax[1, 0].plot(xf[large_peaks1], spec1[large_peaks1], 'b.')
       #ax[1, 0].plot(xf[large_peaks2], spec2[large_peaks2], 'b.')
       #ax[1, 0].plot(xf[small_peaks1], spec1[small_peaks1], 'g.')
       #ax[1, 0].plot(xf[small_peaks2], spec2[small_peaks2], 'g.')
       #ax[1, 0].axhline(0.025)
       #ax[1, 0].axhline(0.25)
       #ax[1, 0].set_xlim(0, 2000)

       #ax[0, 1].plot(time[start:end], wave3[start:end], 'r')
       #ax[0, 1].plot(time[start:end], wave4[start:end], 'k')
       #ax[1, 1].plot(xf, spec3, 'r')
       #ax[1, 1].plot(xf, spec4, 'k')
       #ax[1, 1].plot(xf[large_peaks3], spec3[large_peaks3], 'b.')
       #ax[1, 1].plot(xf[large_peaks4], spec4[large_peaks4], 'b.')
       #ax[1, 1].plot(xf[small_peaks3], spec3[small_peaks3], 'g.')
       #ax[1, 1].plot(xf[small_peaks4], spec4[small_peaks4], 'g.')
       #ax[1, 1].axhline(0.025)
       #ax[1, 1].axhline(0.25)
       #ax[1, 1].set_xlim(0, 2000)
       #plt.show()



        if large_peak1_num * large_peak2_num * large_peak3_num * large_peak4_num:
            peaks1_less = small_peaks1[xf[small_peaks1] < xf[large_peaks1]]
            peaks1_great = small_peaks1[xf[small_peaks1] > xf[large_peaks1]]
            peaks2_less = small_peaks2[xf[small_peaks2] < xf[large_peaks2]]
            peaks2_great = small_peaks2[xf[small_peaks2] > xf[large_peaks2]]
            peaks3_less = small_peaks3[xf[small_peaks3] < xf[large_peaks3]]
            peaks3_great = small_peaks3[xf[small_peaks3] > xf[large_peaks3]]
            peaks4_less = small_peaks4[xf[small_peaks4] < xf[large_peaks4]]
            peaks4_great = small_peaks4[xf[small_peaks4] > xf[large_peaks4]]

            small_peaks1_num_less = np.all(spec1[peaks1_less] <= small_peakl_prom)
            small_peaks2_num_less = np.all(spec2[peaks2_less] <= small_peakl_prom)
            small_peaks3_num_less = np.all(spec3[peaks3_less] <= small_peakl_prom)
            small_peaks4_num_less = np.all(spec4[peaks4_less] <= small_peakl_prom)

            small_peaks1_num_great = len(peaks1_great) == 0
            small_peaks2_num_great = len(peaks2_great) == 0
            small_peaks3_num_great = len(peaks3_great) == 0
            small_peaks4_num_great = len(peaks4_great) == 0

            small_peak1_num = small_peaks1_num_great * small_peaks1_num_less
            small_peak2_num = small_peaks2_num_great * small_peaks2_num_less
            small_peak3_num = small_peaks3_num_great * small_peaks3_num_less
            small_peak4_num = small_peaks4_num_great * small_peaks4_num_less

            if small_peak1_num * small_peak2_num * small_peak3_num * small_peak4_num:
                single_freq.append(t_range)

                #elif div1 * div2 * div3 * div4:
                #    single_freq.append(t_range)
                #    print('div')

    single_freq = np.array(single_freq)

    return single_freq
