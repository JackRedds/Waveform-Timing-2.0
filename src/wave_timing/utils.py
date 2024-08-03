import numpy as np
import pandas as pd
from pytplot import get_data as get_psp_data
import pyspedas
from scipy import signal
from scipy.fft import fft, fftfreq
import wave_timing as wt

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

        vac_data = pyspedas.psp.fields(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l2',
            time_clip=True,
            file_path=self.data_file_path
            )

        mag_data = get_psp_data('psp_fld_l2_mag_SC_4_Sa_per_Cyc')
        date = pd.to_datetime(mag_data.times, unit='s')
        mag_comp = pd.DataFrame(mag_data.y, columns=['Bx', 'By', 'Bz'], index=date)
        B = np.sqrt(mag_comp['Bx'] ** 2 + mag_comp['By'] ** 2 + mag_comp['Bz'] ** 2)

        mag_comp.insert(3, '|B|', B, True)

        return mag_comp


def find_peaks_and_troughs(signal):
    dict = {'distance': 50, 'prominence': 0.5 * np.mean(signal)}
    peaks, _ = signal.find_peaks(signal, **dict)
    troughs = signal.find_peaks(-signal, **dict)

    return peaks, troughs


def trim_delays(delay_12, time_12, delay_34, time_34):
    if len(time_12) > len(time_34):
        length = len(time_34)
        delay_12_new = np.zeros(length)
        time_12_new = np.zeros(length)
        for i in range(length):
            idx = wt.find_nearest(time_12, time_34[i])
            delay_12_new[i] = delay_12[idx]
            time_12_new[i] = time_12[idx]

        delay_12 = delay_12_new
        time_12 = time_12_new

    if len(time_12) < len(time_34):
        length = len(time_12)
        delay_34_new = np.zeros(length)
        time_34_new = np.zeros(length)
        for i in range(length):
            idx = wt.find_nearest(time_34, time_12[i])
            delay_34_new[i] = delay_34[idx]
            time_34_new[i] = time_34[idx]

        delay_34 = delay_34_new
        time_34 = time_34_new


    return delay_12, time_12, delay_34, time_34


def sliding_fft(signal, window_size=5000, hop_size=2500):

    # possibly work on normalization

    signal_length = len(signal)
    samp_rt = wt.sample_rate(signal)
    num_windows = (signal_length - window_size) // hop_size + 1
    spectra = []
    time_range = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + hop_size
        window = signal[start : end] * np.hanning(window_size)
        spectrum = np.abs(fft(window) / window_size)

        spectra.append(spectrum)
        time_range.append([start, end])

    spectra = np.array(spectra)
    time_range = np.array(time_range)

    return spectra, time_range


def peak_to_peak(signal):
    peaks, troughs = find_peaks_and_troughs(signal)
    length = len(peaks)
    ave_peak_to_peak = np.zeros(length)
    for i in range(length):
        trough_less = troughs[troughs < peaks[i]]
        if len(trough_less) > 0:
            peak_less = abs(signal[trough_less[-1]]) + abs(signal[peaks[i]])

        trough_greater = troughs[troughs > peaks[i]]
        if len(trough_greater) > 0:
            peak_greater = abs(signal[trough_greater[0]]) + abs(signal[peaks[i]])

        if len(trough_less) == 0 and len(trough_greater) == 0:
            ave_pk = np.nan

        elif len(trough_less) == 0:
            ave_pk = trough_greater

        elif len(trough_greater) == 0:
            ave_pk = trough_less

        else:
            ave_pk = (peak_less + peak_greater) / 2

        ave_peak_to_peak[i] = ave_pk
    return ave_peak_to_peak


def frequency_filter(wave1, wave2, wave3, wave4):
    spectra1, time_range = sliding_fft(wave1)
    spectra2, __ = sliding_fft(wave2)
    spectra3, __ = sliding_fft(wave3)
    spectra4, __ = sliding_fft(wave4)
    single_freq = []

    for t_range, spec1, spec2, spec3, spec4 in zip(time_range, spectra1, spectra2, spectra3, spectra4):
       large_peaks = {'prominence': 0.2}
       small_peaks = {'prominence': [0.1, 0.2]}

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

       small_peak1_num = len(small_peaks1) == 0
       small_peak2_num = len(small_peaks2) == 0
       small_peak3_num = len(small_peaks3) == 0
       small_peak4_num = len(small_peaks4) == 0

       if large_peak1_num * large_peak2_num * large_peak3_num * large_peak4_num:
           if small_peak1_num * small_peak2_num * small_peak3_num * small_peak4_num:
               single_freq.append(t_range)

    return single_freq
