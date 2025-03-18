import numpy as np
import pandas as pd
from wave_timing.math import geometry, trig, analysis
from wave_timing.signal import operation, calc, delay
from wave_timing.utils import data, directory
from wave_timing.diagnostics import waveform_plot
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib as mpl


file_list_column = [
    [
        sg.Text("Enter Start and End Times:", key="-START END-")
    ],
    [
        sg.Text("Start Time: "),
        sg.InputText(key = "-START-"),
    ],
    [
        sg.Text("End Time: "),
        sg.InputText(key = "-END-"),
    ],
    [
        sg.Button('OK', key="-OK-"),
        sg.Button('Settings', key="-SETTINGS-")
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]


window = sg.Window("Image Viewer", file_list_column)

taps = 601
cutoff = [100, 900]
v1_vec = np.array([1, 0])
v5_vec = np.array([0, 0, 1])
data_path = directory.wave_train_data_dir

Timing_Type = []
Timing_Date = []
Delay_Pos = []
V12_Delay = []
V34_Delay = []
V12_Err = []
V34_Err = []
B_Vec = []
B_Mag = []
SW_Vec = []
SW_Mag = []


while True:
    event, values = window.read()

    if event == "-OK-":
        start_date = values["-START-"]
        end_date = values["-END-"]
        info = data.get_data(start_date, end_date)

        while data.iserror(info.get_vac_data):
            window["-START END-"].update("Invalid Time. Try Again:")
            event, values = window.read()
            info = data.get_data(start_date, end_date)

        dV1, dV2, dV3, dV4, __ = info.get_vac_data()
        time = dV1.index.to_numpy()
        vac_sample_rt, __, dt = calc.sample_rate(time)
        mag_data = info.get_mag_data()
        sw_data = info.get_sw_data()
        filters = trig.FIRBandPass(taps, cutoff, vac_sample_rt)

        dV_trials = dV1.columns

        B_vec_data = mag_data[['Bx', 'By', 'Bz']]
        B_mag_data = mag_data['|B|']
        B_date = mag_data.index

        V_vec_data = sw_data[['Vx', 'Vy', 'Vz']]
        V_mag_data = sw_data['|V|']
        V_date = sw_data.index

        fnames = dV_trials

        window["-FILE LIST-"].update(fnames)


    elif event == "-SETTINGS-":
        settings_list = [
            [
                sg.Text("Input Upper and Lower Frequencies (Default: 100-900 Hz)")
            ],
            [
                sg.Text("Lower Frequency: "),
                sg.InputText(key = "-LWR-FREQ-"),
            ],
            [
                sg.Text("Upper Frequency"),
                sg.InputText(key = "-UPR-FREQ-"),
            ],
            [
                sg.Button('Update', key="-UPDATE-"),
            ]
        ]

        settings_window = sg.Window("Settings", settings_list)

        while True:
            event_3, values_3 = settings_window.read()

            if event_3 == "-UPDATE-DATA-":
                cutoff = [int(values_3["-LWR-FREQ-"]), int(values_3["-UPR-FREQ-"])]
                settings_window.close()
            elif event_3 == "Exit" or event_3 == sg.WINDOW_CLOSED:
                break


    elif event == "-FILE LIST-":
        date = values["-FILE LIST-"][0]

        dv1_unfilt = dV1[date].to_numpy()
        dv2_unfilt = dV2[date].to_numpy()
        dv3_unfilt = dV3[date].to_numpy()
        dv4_unfilt = dV4[date].to_numpy()

        dv1 = filters.filter(dv1_unfilt)
        dv2 = filters.filter(dv2_unfilt)
        dv3 = filters.filter(dv3_unfilt)
        dv4 = filters.filter(dv4_unfilt)

        d_dv1 = analysis.derivitive(dv1, time)
        d_dv2 = analysis.derivitive(dv2, time)
        d_dv3 = analysis.derivitive(dv3, time)
        d_dv4 = analysis.derivitive(dv4, time)

        d_dv1 = filters.filter(d_dv1)
        d_dv2 = filters.filter(d_dv2)
        d_dv3 = filters.filter(d_dv3)
        d_dv4 = filters.filter(d_dv4)
        delay_calc = delay.cross_correlation(dv1, dv2, dv3, dv4, time)

        time_range_chose = [
            [
                sg.Text("Select range you want to time: ", key="-RANGE-")
            ],
            [
                sg.Text("Start Time (s): "),
                sg.InputText(key = "-START-RANGE-")
            ],
            [
                sg.Text("End Time (s): "),
                sg.InputText(key = "-END-RANGE-")
            ],
            [
                sg.Button('PLOT', key="-PLOT-RANGE-", button_color='red'),
                sg.Button('FFT', key="-FFT-RANGE-", button_color='red'),
                sg.Button('Find Single Freq', key="-SINGLE-FREQ-", button_color='red')
            ],
            [
                sg.Button('Train Timing', key="-TRAIN-", button_color='green'),
                sg.Button('Solitary Timing', key="-SOLITARY-", button_color='green'),
            ]
        ]

        time_range_window = sg.Window("Time Range", time_range_chose)

        while True:
            event_2, values_2 = time_range_window.read()

            if event_2 == "-TRAIN-" or event_2 == "-SOLITARY-":
                start_time = float(values_2["-START-RANGE-"])
                end_time = float(values_2["-END-RANGE-"])
                while start_time < time[0] or end_time > time[-1]:
                    time_range_window["-RANGE-"].update("Invalid Time Range. Try Again:")
                    event_2, values_2 = time_range_window.read()
                    start_time = float(values_2["-START-RANGE-"])
                    end_time = float(values_2["-END-RANGE-"])

                delay_pos = (end_time + start_time) / 2

            if event_2 == "Exit" or event_2 == sg.WINDOW_CLOSED:
                plt.close()
                break

            elif event_2 == "-PLOT-RANGE-":
                fig, ax = plt.subplots(2, figsize = (20, 10), sharex=True)
                waveform_plot.plot_waveform(time, ax[0], True, dv1=dv1, dv2=dv2)
                waveform_plot.plot_waveform(time, ax[1], True, dv3=dv3, dv4=dv4)
                ax[1].set_xlabel('Time (s)')
                ax[0].set_xlim(time.min(), time.max())
                plt.tight_layout()
                plt.show(block=False)

            elif event_2 == "-FFT-RANGE-":
                fig, ax = plt.subplots(4, sharex=True, sharey=True, figsize=(20, 10))
                waveform_plot.plot_sliding_fft(time, ax, True, dv1=dv1_unfilt, dv2=dv2_unfilt, dv3=dv3_unfilt, dv4=dv4_unfilt)
                plt.show(block=True)

            elif event_2 == "-SINGLE-FREQ-":
                single_freq_times, __ = operation.frequency_filter(dv1, dv2, dv3, dv4, time)

                fig, ax = plt.subplots(2, figsize = (20, 10), sharex=True)
                ax[0].plot(time, dv1, label='dv1', color='k')
                ax[0].plot(time, dv2, label='dv2', color='r')
                ax[0].legend(loc = 'upper right')
                ax[0].set_ylim(-0.5, 0.5)
                ax[1].plot(time, dv3, label='dv3', color='k')
                ax[1].plot(time, dv4, label='dv4', color='r')
                ax[1].legend(loc = 'upper right')
                ax[1].set_ylim(-0.5, 0.5)

                for times in single_freq_times:
                    ax[0].fill_between(times, -0.5, 0.5, color='grey', alpha=0.4)
                    ax[1].fill_between(times, -0.5, 0.5, color='grey', alpha=0.4)

                ax[1].set_xlabel('Time (s)')
                ax[0].set_xlim(time.min(), time.max())
                plt.tight_layout()
                plt.show(block=False)

            elif event_2 == "-TRAIN-":
                time_range_window.close()
                plt.close()
                timing_delay, timing_corr, __ = delay_calc.find_delay(start_time, end_time)

                Timing_Type.append("Train")
                Timing_Date.append(date)
                V12_Delay.append(timing_delay[0])
                V34_Delay.append(timing_delay[1])
                V12_Err.append(np.nan)
                V34_Err.append(np.nan)
                Delay_Pos.append(delay_pos)
                B_idx = calc.find_nearest(B_date, date)
                SW_idx = calc.find_nearest(V_date, date)
                B_Vec.append(B_vec_data.iloc[B_idx].to_numpy())
                B_Mag.append(B_mag_data.iloc[B_idx])
                SW_Vec.append(V_vec_data.iloc[SW_idx].to_numpy())
                SW_Mag.append(V_mag_data.iloc[SW_idx])

                break

            elif event_2 == "-SOLITARY-":
                time_range_window.close()
                plt.close()
                timing_delay, timing_corr, timing_error = delay_calc.interpolation(start_time, end_time)

                Timing_Type.append("Solitary")
                Timing_Date.append(date)
                V12_Delay.append(timing_delay[0])
                V34_Delay.append(timing_delay[1])
                V12_Err.append(timing_error[[0]])
                V34_Err.append(timing_error[1])
                Delay_Pos.append(delay_pos)
                B_idx = calc.find_nearest(B_date, date)
                SW_idx = calc.find_nearest(V_date, date)
                B_Vec.append(B_vec_data.iloc[B_idx].to_numpy())
                B_Mag.append(B_mag_data.iloc[B_idx])
                SW_Vec.append(V_vec_data.iloc[SW_idx].to_numpy())
                SW_Mag.append(V_mag_data.iloc[SW_idx])

                break

    elif event == "Exit" or event == sg.WINDOW_CLOSED:
        break

timing_type = np.array(Timing_Type)
timing_date = np.array(Timing_Date)
v12_delay = np.array(V12_Delay)
v34_delay = np.array(V34_Delay)
v12_err = np.array(V12_Err)
v34_err = np.array(V34_Err)
delay_pos = np.array(Delay_Pos)
B_vec = np.array(B_Vec)
B_mag = np.array(B_Mag)
V_vec = np.array(SW_Vec)
V_mag = np.array(SW_Mag)

d = {f'Delay Time [s after Date]': delay_pos, 'v12 Delay sc [s]': v12_delay,
     'v34 Delay sc [s]': v34_delay} #Add wave frequency

wave_properties = pd.DataFrame(d, index=timing_date)

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
wave_properties['|B| xy [nT]'] = np.sqrt(geometry.dot_product(B_vec[:, :-1], B_vec[:, :-1]))

# Magnetic Field Magnitude
wave_properties['|B| [nT]'] = np.sqrt(geometry.dot_product(B_vec, B_vec))

data.save_data(wave_properties, data_path, f"{date}.csv")
