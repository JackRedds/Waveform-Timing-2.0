import numpy as np
import pandas as pd
from scipy import signal
from wave_timing.math import trig
from wave_timing.signal import calc


def find_peaks_and_troughs(sig: np.ndarray):
    assert len(sig) != 0

    dict = {'distance': 50, 'prominence': 0.5 * np.mean(sig)}
    peaks = signal.find_peaks(sig, **dict)[0]
    troughs = signal.find_peaks(-sig, **dict)[0]

    return peaks, troughs


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

    dt = calc.sample_rate(time)[2]
    spectra1, __, xf, time_range = trig.sliding_fft(wave1, time, window_size=window_size, hop_size=hop_size)
    time_range = np.array(time_range)
    spectra2 = trig.sliding_fft(wave2, time, window_size=window_size, hop_size=hop_size)[0]
    spectra3 = trig.sliding_fft(wave3, time, window_size=window_size, hop_size=hop_size)[0]
    spectra4 = trig.sliding_fft(wave4, time, window_size=window_size, hop_size=hop_size)[0]
    single_freq = []
    wave_freq = []

    for t_range, spec1, spec2, spec3, spec4 in zip(time_range, spectra1, spectra2, spectra3, spectra4):
        large_peaks = {'prominence': large_peak_prom}
        small_peaks = {'prominence': [small_peakg_prom, large_peak_prom]}

        large_peaks1 = signal.find_peaks(spec1, **large_peaks)[0]
        large_peaks2 = signal.find_peaks(spec2, **large_peaks)[0]
        large_peaks3 = signal.find_peaks(spec3, **large_peaks)[0]
        large_peaks4 = signal.find_peaks(spec4, **large_peaks)[0]

        small_peaks1 = signal.find_peaks(spec1, **small_peaks)[0]
        small_peaks2 = signal.find_peaks(spec2, **small_peaks)[0]
        small_peaks3 = signal.find_peaks(spec3, **small_peaks)[0]
        small_peaks4 = signal.find_peaks(spec4, **small_peaks)[0]

        large_peak1_num = len(large_peaks1) == 1
        large_peak2_num = len(large_peaks2) == 1
        large_peak3_num = len(large_peaks3) == 1
        large_peak4_num = len(large_peaks4) == 1

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
                wave_freq.extend(xf[large_peaks1])

    single_freq = np.array(single_freq) * dt
    wave_freq = np.array(wave_freq)
    return single_freq, wave_freq


def B_V_vec_near_time(trial_date, B_data, V_data, delay_pos):
    B_vec_data = B_data[['Bx', 'By', 'Bz']]
    B_date = B_data.index

    V_vec_data = V_data[['Vx', 'Vy', 'Vz']]
    V_date = V_data.index

    B_vec = []
    V_vec = []
    delay_date = []

    for t in delay_pos:
        date = trial_date + pd.Timedelta(seconds=t)
        delay_date.append(date)

        B_index = calc.find_nearest(B_date, date)
        b_vec = B_vec_data.iloc[B_index]
        B_vec.append(b_vec)

        V_index = calc.find_nearest(V_date, date)
        v_vec = V_vec_data.iloc[V_index]
        V_vec.append(v_vec)

    delay_date = np.array(delay_date)
    B_vec = np.array(B_vec)
    V_vec = np.array(V_vec)

    return B_vec, V_vec, delay_date
