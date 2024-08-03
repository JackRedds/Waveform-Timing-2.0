import numpy as np
import pandas as pd
from pandas.tseries import frequencies
from scipy import signal, interpolate
from astropy import units as u

def find_nearest(signal, value):
    # TODO


def sample_rate(signal):
    # TODO


class cross_correlation:

    def __init__(self, wave1, wave2, wave3, wave4, time):
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave3 = wave3
        self.wave4 = wave4
        self.time = time


    def x_corr(self):
        wave1 = self.wave1
        wave2 = self.wave2
        norm1 = (wave1 - np.mean(wave1)) / (np.std(wave1) * len(wave1))
        norm2 = (wave2 - np.mean(wave2)) / np.std(wave2)

        wave3 = self.wave3
        wave4 = self.wave4
        norm3 = (wave3 - np.mean(wave3)) / (np.std(wave3) * len(wave3))
        norm4 = (wave4 - np.mean(wave4)) / np.std(wave4)

        lags = signal.correlation_lags(len(norm1), len(norm2), mode='full')

        correlation_12 = signal.correlate(norm1, norm2, mode='full', method='fft')
        xmax_12 = lags[np.argmax(correlation_12)]

        correlation_34 = signal.correlate(norm3, norm4, mode='full', method='fft')
        xmax_34 = lags[np.argmax(correlation_34)]

        return [xmax_12, xmax_34], [correlation_12, correlation_34], lags


    def interpolation(self, interp=10):
        f1 = interpolate.interp1d(self.time, self.wave1, fill_value="extrapolate", kind="quadratic")
        f2 = interpolate.interp1d(self.time, self.wave2, fill_value="extrapolate", kind="quadratic")
        f3 = interpolate.interp1d(self.time, self.wave3, fill_value="extrapolate", kind="quadratic")
        f4 = interpolate.interp1d(self.time, self.wave4, fill_value="extrapolate", kind="quadratic")

        time_interp = np.linspace(self.time[0], self.time[-1], len(self.time) * interp)
        self.wave1 = f1(time_interp)
        self.wave2 = f2(time_interp)
        self.wave3 = f3(time_interp)
        self.wave4 = f4(time_interp)

        xmax, correlation, lags = self.x_corr()
        pks_12, _ = signal.find_peaks(-correlation[0])
        pks_34, _ = signal.find_peaks(-correlation[1])

        def error_determination(pks, correlation, lags):
            if len(pks) > 1:
                upr = find_nearest(pks[pks>np.argmax(correlation)], np.argmax(correlation))
                lwr = find_nearest(pks[pks<np.argmax(correlation)], np.argmax(correlation))
                upr = pks[pks>np.argmax(correlation)][upr]
                lwr = pks[pks<np.argmax(correlation)][lwr]
                correlation = correlation[lwr:upr]
                lags = lags[lwr:upr]

            err_det = 1.2 * (1 - correlation[np.argmax(correlation)])
            vals = lags[(1 - correlation) < err_det]
            lwr_bnd = vals.min()
            upr_bnd = vals.max()
            upr_err = abs(lags[np.argmax(correlation)] - upr_bnd)
            lwr_err = abs(lags[np.argmax(correlation)] - lwr_bnd)
            err = (lwr_err + upr_err) / 2


        err_12 = error_determination(pks_12, correlation[0], lags)
        err_34 = error_determination(pks_34, correlation[1], lags)

        return xmax, correlation, [err_12, err_34]


    def find_delay(self, peak_width=5):
        samp_rt = sample_rate(self.wave1)

        def find_phase(corr, fs, peak_width):
            corr_max = signal.find_peaks(corr, width=peak_width)[0]

            if len(corr_max) > 5:
                n_per_cycle = np.mean(np.diff(corr_max)[2:-2])
            else:
                n_per_cycle = np.mean(np.diff(corr_max))
            period = n_per_cycle / fs
            frequency = 1.0 / period
            shifts = corr_max - (len(corr) // 2)
            i_delay = np.argmin(np.abs(shifts))
            fract = shifts[i_delay] / n_per_cycle
            phase = fract * 360.0
            delay = fract * period
            return phase, frequency, delay

        xmax, correlation, lags = self.x_corr()

        __, frequency_12, delay_12 = find_phase(correlation[0], samp_rt, peak_width)
        __, frequency_34, delay_34 = find_phase(correlation[1], samp_rt, peak_width)

        corr_max_12 = np.max(correlation[0])
        corr_max_34 = np.max(correlation[1])

        return [delay_12, delay_34], [corr_max_12, corr_max_34], [frequency_12, frequency_34]


    def n_x_corr(self, window_size=5000, hop_size=2500):
        signal_len = len(self.wave1)
        num_windows = (signal_len - window_size) // hop_size + 1
        time_delays = np.zeros((2, num_windows))


        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            self.wave1 = self.wave1[start : end]
            self.wave2 = self.wave2[start : end]
            self.wave3 = self.wave3[start : end]
            self.wave4 = self.wave4[start : end]
            time_delays[i], correlation, _ = self.x_corr()

            corr_max_12 = correlation[0].max()
            corr_max_34 = correlation[1].max()

            if corr_max_34 < 0.5 or corr_max_34 < 0.5:
                time_delays[i] = [np.nan, np.nan]

        return time_delays


def derivitive(signal, time):
    length = len(signal)
    derivitive = np.zeros(length)
    td = time[1] - time[0]

    for i in range(length - 1):
        derivitive[i] = signal[i + 1] - signal[i]

    return derivitive


def angle_between(vec1, vec2):
    dot = np.dot(vec1, vec2)
    mag1 = np.sqrt(np.dot(vec1, vec1))
    mag2 = np.sqrt(np.dot(vec2, vec2))
    angle = np.arccos(dot / (mag1 * mag2))
    angle = np.degrees(angle)
    return angle


def delay(signal1, signal2):
    if len(signal1) > len(signal2):
        length = len(signal2)
        delays = np.zeros(length)
        ave_position = np.zeros(length)
        for i in range(length):
            idx = find_nearest(signal1, signal2[i])
            delays[i] = signal2[i] - signal1[idx]
            ave_position[i] = (signal1[idx] + signal2[i]) / 2

    elif len(signal1) <= len(signal2):
        length = len(signal1)
        delays = np.zeros(length)
        ave_position = np.zeros(length)
        for i in range(length):
            idx = find_nearest(signal2, signal1[i])
            delays[i] = signal2[idx] - signal1[i]
            ave_position = (signal1[i] + signal2[idx]) / 2

    return delays, ave_position


def wave_velocity(delay_12, delay_34, boom_length=3.5*u.m):
    delay_vector = np.array([delay_12, delay_34]) * u.s
    delay_mag = np.sqrt(np.dot(delay_vector, delay_vector))
    velocity = boom_length / delay_mag

    k_vector = velocity * delay_vector / boom_length
    velocity = velocity.to(u.km / u.s)
    return velocity, k_vector
