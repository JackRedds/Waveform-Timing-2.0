import numpy as np
import pandas as pd
from wave_timing.math import geometry, trig, analysis
from wave_timing.signal import operation, calc, delay
from wave_timing.utils import data


start_date = '2021-01-19'
end_date = '2021-01-20'
taps = 601
cutoff = [100, 900]
v1_vec = np.array([1, 0])
v5_vec = np.array([0, 0, 1])

data = data.get_data(start_date, end_date)
dV1, dV2, dV3, dV4, __ = data.get_vac_data()
time = dV1.index.to_numpy()
vac_sample_rt, __, dt = calc.sample_rate(time)
mag_data = data.get_mag_data()
sw_data = data.get_sw_data()
filters = trig.FIRBandPass(taps, cutoff, vac_sample_rt)

for date in dV1.columns:
    end_date = date + pd.Timedelta(time[-1], unit='s')
    dv1 = dV1[date].to_numpy()
    dv2 = dV2[date].to_numpy()
    dv3 = dV3[date].to_numpy()
    dv4 = dV4[date].to_numpy()

    dv1 = filters.filter(dv1)
    dv2 = filters.filter(dv2)
    dv3 = filters.filter(dv3)
    dv4 = filters.filter(dv4)

    d_dv1 = analysis.derivitive(dv1, time)
    d_dv2 = analysis.derivitive(dv2, time)
    d_dv3 = analysis.derivitive(dv3, time)
    d_dv4 = analysis.derivitive(dv4, time)

    d_dv1 = filters.filter(d_dv1)
    d_dv2 = filters.filter(d_dv2)
    d_dv3 = filters.filter(d_dv3)
    d_dv4 = filters.filter(d_dv4)

    single_freq_times, wave_freq = operation.frequency_filter(dv1, dv2, dv3, dv4, time)
    delays = delay.multiple_delay(dv1, dv2, dv3, dv4, time)

    v12_delay = []
    v34_delay = []
    delay_pos = []
    wave_frequency = []

    for t_range, freq in zip(single_freq_times, wave_freq):
        d_peaks, d_pos_peaks, d_troughs, d_pos_troughs = delays.calculate_delays(start=t_range[0], end=t_range[1])
        v12_delay.extend(d_peaks[0])
        v34_delay.extend(d_peaks[1])
        delay_pos.extend(d_pos_peaks)
        v12_delay.extend(d_troughs[0])
        v34_delay.extend(d_troughs[1])
        delay_pos.extend(d_pos_troughs)
        length = len(d_pos_peaks) + len(d_pos_troughs)
        wave_frequency.extend(np.full(length, freq))


    v12_delay = np.array(v12_delay)
    v34_delay = np.array(v34_delay)
    delay_pos = np.array(delay_pos)
    wave_frequency = np.array(wave_frequency)

    B_vec, V_vec, delay_date = operation.B_V_vec_near_time(date, mag_data, sw_data, delay_pos)

    d = {f'Delay Time [s after {date}]': delay_pos, 'v12 Delay sc [s]': v12_delay,
         'v34 Delay sc [s]': v34_delay, 'Wave Frequency sc [Hz]': wave_frequency}

    wave_properties = pd.DataFrame(d, index=delay_date)


    # Wave velocity Space Craft Frame
    wave_velocity_sc = analysis.wave_velocity(v12_delay, v34_delay)
    wave_properties[['Vp sc v12 [km/s]', 'Vp sc v34 [km/s]']] = wave_velocity_sc

    # Wave Velocity Magnitude Space Craft Frame
    wave_properties['|Vp| sc [km/s]'] = np.sqrt(geometry.dot_product(wave_velocity_sc, wave_velocity_sc))

    # Angle Between Wave and V1 Boom Space Craft Frame
    wave_properties['Vp-V1 sc Angle [Deg]'] = geometry.angle_between(wave_velocity_sc, v1_vec)

    # Angle Between Wave and Solar Wind Space Craft Frame
    wave_properties['Vp-Vsw sc Angle [Deg]'] = geometry.angle_between(wave_velocity_sc, V_vec[:, :-1])

    # Wave Velocity Plasma Frame
    wave_velocity_pf = wave_velocity_sc - V_vec[:, :-1]
    wave_properties[['Vp pf v12 [km/s]', 'Vp pf v12 [km/s]']] = wave_velocity_pf

    # Wave Velocity Magnitude Plasma Frame
    wave_properties['|Vp| pf [km/s]'] = np.sqrt(geometry.dot_product(wave_velocity_pf, wave_velocity_pf))

    # Angle Between Wave and Magnetic Field Plasma Frame
    wave_properties['Vp-B pf Angle [Deg]'] = geometry.angle_between(B_vec[:, :-1], wave_velocity_pf)

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
    wave_properties['|B| [nT]'] = np.sqrt(geometry.dot_product(B_vec[:, :-1], B_vec[:, :-1]))

    # Magnetic Field Magnitude
    wave_properties['|B| [nT]'] = np.sqrt(geometry.dot_product(B_vec, B_vec))

    print(wave_properties)
    exit()
