import wave_timing.utils as wtu
import wave_timing.calc as wtc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


start_date = '2021-01-19'
end_date = '2021-01-20'
taps = 601
cutoff = [100, 900]

data = wtu.get_data(start_date, end_date)
dV1, dV2, dV3, dV4, __ = data.get_vac_data()
time = dV1.index.to_numpy()
vac_sample_rt, __, dt = wtc.sample_rate(time)
mag_data = data.get_mag_data()
sw_data = data.get_sw_data()
filters = wtu.FIRBandPass(taps, cutoff, vac_sample_rt)

B_vec_data = mag_data[['Bx', 'By', 'Bz']]
B_mag_data = mag_data['|B|']
B_date = mag_data.index

V_vec_data = sw_data[['Vx', 'Vy', 'Vz']]
V_mag_data = sw_data['|V|']
V_date = sw_data.index

date = dV1.columns[4]

dv1 = dV1[date].to_numpy()
d_dv1 = wtc.derivitive(dv1, time)
dv2 = dV2[date].to_numpy()
d_dv2 = wtc.derivitive(dv2, time)
dv3 = dV3[date].to_numpy()
d_dv3 = wtc.derivitive(dv3, time)
dv4 = dV4[date].to_numpy()
d_dv4 = wtc.derivitive(dv4, time)

dv1 = filters.filter(dv1)
d_dv1 = filters.filter(d_dv1)
dv2 = filters.filter(dv2)
d_dv2 = filters.filter(d_dv2)
dv3 = filters.filter(dv3)
d_dv3 = filters.filter(d_dv3)
dv4 = filters.filter(dv4)
d_dv4 = filters.filter(d_dv4)

B_index = wtc.find_nearest(B_date, date)
V_index = wtc.find_nearest(V_date, date)

B_vec = B_vec_data.iloc[B_index]
B_mag = B_mag_data.iloc[B_index]

V_vec = V_vec_data.iloc[V_index]
V_mag = V_mag_data.iloc[V_index]

single_freq_times = wtu.frequency_filter(dv1, dv2, dv3, dv4, time) * dt
delays = wtc.multiple_delay(dv1, dv2, dv3, dv4, time)

v12_delay_peaks = []
v34_delay_peaks = []
v12_delay_peaks_pos = []
v34_delay_peaks_pos = []
v12_delay_troughs = []
v34_delay_troughs = []
v12_delay_troughs_pos = []
v34_delay_troughs_pos = []

for start, end in single_freq_times:
    d_peaks, d_peaks_pos, d_troughs, d_troughs_pos = delays.calculate_delays(start=start, end=end)
    v12_delay_peaks.extend(d_peaks[0])
    v34_delay_peaks.extend(d_peaks[1])
    v12_delay_peaks_pos.extend(d_peaks_pos[0])
    v34_delay_peaks_pos.extend(d_peaks_pos[1])
    v12_delay_troughs.extend(d_troughs[0])
    v34_delay_troughs.extend(d_troughs[1])
    v12_delay_troughs_pos.extend(d_troughs_pos[0])
    v34_delay_troughs_pos.extend(d_troughs_pos[1])

fig, ax = plt.subplots(2, sharex=True, figsize=(10, 5))
ax[0].plot(time, dv1, 'r')
ax[0].plot(time, dv2, 'g')
ax[0].plot(v12_delay_peaks_pos, v12_delay_peaks, 'b.')
ax[0].plot(v12_delay_troughs_pos, v12_delay_troughs, 'k.')

ax[1].plot(time, dv3, 'r')
ax[1].plot(time, dv4, 'g')
ax[1].plot(v34_delay_peaks_pos, v34_delay_peaks, 'b.')
ax[1].plot(v34_delay_troughs_pos, v34_delay_troughs, 'k.')
plt.show()
