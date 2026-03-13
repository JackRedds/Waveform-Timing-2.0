import numpy as np
import pandas as pd
from wave_timing.math import geometry, trig, analysis
from wave_timing.signal import operation, calc, delay
from wave_timing.utils import data, directory
from wave_timing.diagnostics import waveform_plot
import matplotlib.pyplot as plt

#Have you run than angle code for the VDC data from 2022-12-10 21:00 to 24:00?
#Also 2021-11-22 04:00 to 15:00 and 2023-06-2119:30 to 21:00.


start_date_int = '2023-06-21 19:30'
end_date_int = '2023-06-21 21:00'
taps = 601
cutoff = [100, 900]
v1_vec = np.array([1, 0])
v5_vec = np.array([0, 0, 1])
angle = 90
data_path = directory.b0_in_v1234_data_dir

info = data.get_data(start_date_int, end_date_int)
dV1, dV2, dV3, dV4, __ = info.get_vdc_data()
time = dV1.index.to_numpy()
vac_sample_rt, __, dt = calc.sample_rate(time)
mag_data = info.get_mag_data()
sw_data = info.get_sw_data()

B_data = []
V_data = []
start_date_arr = []
end_date_arr = []

for date in dV1.columns:
    end_date = date + pd.Timedelta(time[-1], unit='s')
    B_vec = np.array(mag_data[date:end_date][['Bx', 'By', 'Bz']].mean())
    V_vec = np.array(sw_data[date:end_date][['Vx', 'Vy', 'Vz']].mean())

    start_date_arr.append(date)
    end_date_arr.append(end_date)
    B_data.append(B_vec)
    V_data.append(V_vec)

start_date = np.array(start_date_arr)
end_date = np.array(end_date_arr)
B_vec = np.array(B_data)
V_vec = np.array(V_data)

d = {'WFC Start Time': date, 'WFC End Time': end_date}

wave_properties = pd.DataFrame(d)

# Angle Between Solar Wind and V1 Boom Space Craft Frame
wave_properties['Vsw-V1 sc Angle [Deg]'] = geometry.angle_between(V_vec[:, :-1], v1_vec)

# Angle Between Solar wind and V5 Boom Space Craft Frame
wave_properties['Vsw-V5 sc Angle [Deg]'] = geometry.angle_between(V_vec, v5_vec)

# Angle Between Magnetic Field and Solar Wind Space Craft Frame
wave_properties['Vsw-B sc Angle [Deg]'] = geometry.angle_between(B_vec, V_vec)

# Angle Between Magnetic Field and Solar Wind in X-Y plane Space Craft Frame
wave_properties['Vsw-B sc xy [Deg]'] = geometry.angle_between(B_vec[:, :-1], V_vec[:, :-1])

# Angle Between Magnetic Field and V1 Boom
wave_properties['B-V1 Angle [Deg]'] = geometry.angle_between(B_vec[:, :-1], v1_vec)

# Angle Between Magnetic Filed and V5 Boom
wave_properties['B-V5 Angle [Deg]'] = geometry.angle_between(B_vec, v5_vec)

# Solar Wind Velocity Vector Space Craft Frame
wave_properties[['Vswx sc [km/s]', 'Vswy sc [km/s]', 'Vswz sc [km/s]']] = V_vec

# Solar Wind Velocity Magnitude in X-Y plane Space Craft Frame
wave_properties['|Vsw| sc xy [km/s]'] = np.sqrt(geometry.dot_product(V_vec[:, :-1], V_vec[:, :-1]))

# Solar wind Velocity Magnitude Space Craft Frame
wave_properties['|Vsw| sc [km/s]'] = np.sqrt(geometry.dot_product(V_vec, V_vec))

# Magnetic Field Vector
wave_properties[['Bx [nT]', 'By [nT]', 'Bz [nT]']] = B_vec

# Magnetic Field Magnitude in X-Y
wave_properties['|B| xy [nT]'] = np.sqrt(geometry.dot_product(B_vec[:, :-1], B_vec[:, :-1]))

# Magnetic Field Magnitude
wave_properties['|B| [nT]'] = np.sqrt(geometry.dot_product(B_vec, B_vec))

wave_properties_within_45 = wave_properties[abs(wave_properties['B-V5 Angle [Deg]']-90) < angle]


if len(wave_properties_within_45) > 0:
    data.save_data(wave_properties_within_45, data_path, f"{start_date_int}_{end_date_int}_within_{angle}.csv")
