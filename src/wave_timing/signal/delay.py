import numpy as np
from scipy import signal, interpolate
from wave_timing.math import analysis
from wave_timing.signal import calc, operation


class cross_correlation:

    def __init__(
            self,
            wave1: np.ndarray,
            wave2: np.ndarray,
            wave3: np.ndarray,
            wave4: np.ndarray,
            time: np.ndarray,
            trim_wave=True
    ):
        # waves probably have to be numpy arrays
        assert len(wave1) != 0
        assert len(wave2) != 0
        assert len(wave3) != 0
        assert len(wave4) != 0
        assert len(time) != 0

        self.wave1 = wave1
        self.wave2 = wave2
        self.wave3 = wave3
        self.wave4 = wave4
        self.dt = calc.sample_rate(time)[2]
        self.time = time
        self.trim_wave = trim_wave


    def x_corr(self, start=0.0, end=3.5, trim=True):
        dt = self.dt
        wave1 = self.wave1
        wave2 = self.wave2
        wave3 = self.wave3
        wave4 = self.wave4
        time = self.time
        start = float(start)
        end = float(end)
        if trim * self.trim_wave:
            start_idx = int(start / dt)
            end_idx = int(end / dt)
            wave1 = wave1[start_idx:end_idx]
            wave2 = wave2[start_idx:end_idx]
            wave3 = wave3[start_idx:end_idx]
            wave4 = wave4[start_idx:end_idx]
            time = time[start_idx:end_idx]

        norm1 = analysis.wave_normalization(wave1)
        norm2 = analysis.wave_normalization(wave2)

        norm3 = analysis.wave_normalization(wave3)
        norm4 = analysis.wave_normalization(wave4)

        lags = signal.correlation_lags(len(norm1), len(norm2), mode='full')

        correlation_12 = signal.correlate(norm1, norm2, mode='full', method='fft') / (len(wave1) - 1)
        xmax_12 = lags[np.argmax(correlation_12)]

        correlation_34 = signal.correlate(norm3, norm4, mode='full', method='fft') / (len(wave3) -1)
        xmax_34 = lags[np.argmax(correlation_34)]

        xmax = np.array([xmax_12, xmax_34])
        correlation = np.array([correlation_12, correlation_34])

        return xmax, correlation, lags


    def interpolation(self, start=0.0, end=3.5, interp=10):
        assert interp > 0
        dt = self.dt
        wave1 = self.wave1
        wave2 = self.wave2
        wave3 = self.wave3
        wave4 = self.wave4
        time = self.time
        start = float(start)
        end = float(end)
        if self.trim_wave:
            start_idx = int(start / dt)
            end_idx = int(end / dt)
            wave1 = wave1[start_idx:end_idx]
            wave2 = wave2[start_idx:end_idx]
            wave3 = wave3[start_idx:end_idx]
            wave4 = wave4[start_idx:end_idx]
            time = time[start_idx:end_idx]

        f1 = interpolate.interp1d(time, wave1, fill_value="extrapolate", kind="quadratic")
        f2 = interpolate.interp1d(time, wave2, fill_value="extrapolate", kind="quadratic")
        f3 = interpolate.interp1d(time, wave3, fill_value="extrapolate", kind="quadratic")
        f4 = interpolate.interp1d(time, wave4, fill_value="extrapolate", kind="quadratic")

        time_interp = np.linspace(time[0], time[-1], len(time) * interp)
        self.wave1 = f1(time_interp)
        self.wave2 = f2(time_interp)
        self.wave3 = f3(time_interp)
        self.wave4 = f4(time_interp)
        self.time = time_interp

        xmax, correlation, lags = self.x_corr(trim=False)
        pks_12, _ = signal.find_peaks(-correlation[0])
        pks_34, _ = signal.find_peaks(-correlation[1])

        def error_determination(pks, correlation, lags):
            if len(pks) > 1:
                upr = calc.find_nearest(pks[pks>np.argmax(correlation)], np.argmax(correlation))
                lwr = calc.find_nearest(pks[pks<np.argmax(correlation)], np.argmax(correlation))
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

            return err

        corr_max_12 = np.max(correlation[0])
        corr_max_34 = np.max(correlation[1])

        err_12 = error_determination(pks_12, correlation[0], lags)
        err_34 = error_determination(pks_34, correlation[1], lags)

        corr_max = np.array([corr_max_12, corr_max_34])
        err = np.array([err_12, err_34])


        return xmax * dt, corr_max, err * dt


    def find_delay(self, start=0.0, end=3.5, peak_width=5):
        assert peak_width > 0
        samp_rt = calc.sample_rate(self.time)[0]
        dt = self.dt
        wave1 = self.wave1
        wave2 = self.wave2
        wave3 = self.wave3
        wave4 = self.wave4
        time = self.time
        start = float(start)
        end = float(end)
        if self.trim_wave:
            start_idx = int(start / dt)
            end_idx = int(end / dt)
            self.wave1 = wave1[start_idx:end_idx]
            self.wave2 = wave2[start_idx:end_idx]
            self.wave3 = wave3[start_idx:end_idx]
            self.wave4 = wave4[start_idx:end_idx]
            self.time = time[start_idx:end_idx]

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

        xmax, correlation, lags = self.x_corr(trim=False)

        __, frequency_12, delay_12 = find_phase(correlation[0], samp_rt, peak_width)
        __, frequency_34, delay_34 = find_phase(correlation[1], samp_rt, peak_width)

        corr_max_12 = np.max(correlation[0])
        corr_max_34 = np.max(correlation[1])

        delay = np.array([delay_12, delay_34])
        corr_max = np.array([corr_max_12, corr_max_34])
        frequency = np.array([frequency_12, frequency_34])
        return delay, corr_max, frequency


    def n_x_corr(self, window_size=50000, hop_size=2500):
        assert window_size > 0
        assert hop_size > 0
        signal_len = len(self.wave1)
        dt = calc.sample_rate(self.time)[2]
        num_windows = (signal_len - window_size) // hop_size + 1
        time_delays = np.zeros((num_windows, 2))
        time = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            time_delays[i], correlation, _ = self.x_corr(start=start * dt, end=end * dt)
            time[i] = (start + end) / 2 * dt

            corr_max_12 = correlation[0].max()
            corr_max_34 = correlation[1].max()

            if corr_max_12 < 0.5 or corr_max_34 < 0.5:
                time_delays[i] = [np.nan, np.nan]

        return time_delays, time


class multiple_delay:
    def __init__(self,
                 wave1: np.ndarray,
                 wave2: np.ndarray,
                 wave3: np.ndarray,
                 wave4: np.ndarray,
                 time: np.ndarray
    ):
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave3 = wave3
        self.wave4 = wave4
        self.time = time
        self.dt = calc.sample_rate(time)[2]


    def peaks_and_troughs(self, start=0.0, end=3.5):
        start = int(start / self.dt)
        end = int(end / self.dt)
        peaks1, troughs1 = operation.find_peaks_and_troughs(self.wave1[start:end])
        peaks2, troughs2 = operation.find_peaks_and_troughs(self.wave2[start:end])
        peaks3, troughs3 = operation.find_peaks_and_troughs(self.wave3[start:end])
        peaks4, troughs4 = operation.find_peaks_and_troughs(self.wave4[start:end])

        peaks = [peaks1, peaks2, peaks3, peaks4]
        troughs = [troughs1, troughs2, troughs3, troughs4]
        return peaks, troughs


    def delay(self, signal1, signal2):
        assert len(signal1) != 0
        assert len(signal2) != 0

        if len(signal1) > len(signal2):
            length = len(signal2)
            delays = np.zeros(length)
            ave_position = np.zeros(length)
            for i in range(length):
                idx = calc.find_nearest(signal1, signal2[i])
                delays[i] = signal2[i] - signal1[idx]
                ave_position[i] = (signal1[idx] + signal2[i]) / 2

        elif len(signal1) <= len(signal2):
            length = len(signal1)
            delays = np.zeros(length)
            ave_position = np.zeros(length)
            for i in range(length):
                idx = calc.find_nearest(signal2, signal1[i])
                delays[i] = signal2[idx] - signal1[i]
                ave_position[i] = (signal1[i] + signal2[idx]) / 2

        return delays, ave_position


    def trim_delays(self, delay_12, time_12, delay_34, time_34):
        assert len(delay_12) != 0
        assert len(delay_34) != 0

        def trim(delay, time):
            length = len(time[0])
            delay_new = np.zeros(length)
            time_new = np.zeros(length)
            for i in range(length):
                idx = calc.find_nearest(time[1], time[0][i])
                delay_new[i] = delay[1][idx]
                time_new[i] = time[1][idx]
            delay[1] = delay_new
            time[1] = time_new
            return delay, time

        if len(time_12) > len(time_34):
            delay = [delay_34, delay_12]
            time = [time_34, time_12]
            delay, time = trim(delay, time)
            temp = delay[0]
            delay[0] = delay[1]
            delay[1] = temp
            temp = time[0]
            time[0] = time[1]
            time[1] = temp

        elif len(time_12) < len(time_34):
            delay = [delay_12, delay_34]
            time = [time_12, time_34]
            delay, time = trim(delay, time)

        else:
            delay = [delay_12, delay_34]
            time = [time_12, time_34]

        assert len(delay[0]) == len(delay[0])

        return delay, time


    def calculate_delays(self, start=0.0, end=3.5):
        dt = self.dt
        peaks, troughs = self.peaks_and_troughs(start=start, end=end)
        peaks_delay_12, peaks_delay_pos_12 = self.delay(peaks[1], peaks[0])
        troughs_delay_12, troughs_delay_pos_12 = self.delay(troughs[1], troughs[0])
        peaks_delay_34, peaks_delay_pos_34 = self.delay(peaks[3], peaks[2])
        troughs_delay_34, troughs_delay_pos_34 = self.delay(troughs[3], troughs[2])

        delay_peaks, delay_pos_peaks = self.trim_delays(peaks_delay_12, peaks_delay_pos_12, peaks_delay_34, peaks_delay_pos_34)
        delay_troughs, delay_pos_troughs = self.trim_delays(troughs_delay_12, troughs_delay_pos_12, troughs_delay_34, troughs_delay_pos_34)

        delay_peaks = np.array(delay_peaks) * dt
        delay_pos_peaks = np.array(delay_pos_peaks) * dt + start
        delay_pos_peaks = np.mean(delay_pos_peaks, axis=0)
        delay_troughs = np.array(delay_troughs) * dt
        delay_pos_troughs = np.array(delay_pos_troughs) * dt + start
        delay_pos_troughs = np.mean(delay_pos_troughs, axis=0)

        return delay_peaks, delay_pos_peaks, delay_troughs, delay_pos_troughs
