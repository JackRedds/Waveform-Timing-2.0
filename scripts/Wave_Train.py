import wave_timing.utils as wtu
import wave_timing.calc as wtc
import numpy as np
import pandas as pd


start_date = '2021-01-19'
end_date = '2021-01-20'
taps = 601
cutoff = [100, 900]

data = wtu.get_data(start_date, end_date)
dV1, dV2, dV3, dV4, __ = data.get_vac_data()
time = dV1.index.to_numpy()
vac_sample_rt = wtc.sample_rate(time)[0]
mag_data = data.get_mag_data()
sw_data = data.get_sw_data()
filters = wtu.FIRBandPass(taps, cutoff, vac_sample_rt)

B_vec_data = mag_data[['Bx', 'By', 'Bz']]
B_mag_data = mag_data['|B|']
B_date = mag_data.index

V_vec_data = sw_data[['Vx', 'Vy', 'Vz']]
V_mag_data = sw_data['|V|']
V_date = sw_data.index

for date in dV1.columns:
    dv1 = dV1[date].to_numpy()
    dv2 = dV2[date].to_numpy()
    dv3 = dV3[date].to_numpy()
    dv4 = dV4[date].to_numpy()

    dv1 = filters.filter(dv1)
    dv2 = filters.filter(dv2)
    dv3 = filters.filter(dv3)
    dv4 = filters.filter(dv4)

    B_index = wtc.find_nearest(B_date, date)
    V_index = wtc.find_nearest(V_date, date)

    B_vec = B_vec_data.iloc[B_index]
    B_mag = B_mag_data.iloc[B_index]

    print(B_date[B_index])
    print(B_vec_data.index[B_index])
    V_vec = V_vec_data.iloc[V_index]
    V_mag = V_mag_data.iloc[V_index]
