import wave_timing.utils as wtu
import wave_timing.calc as wtc
import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 3.5, 6.67e-6)
v12_delay = 0.0
v34_delay = 0.0
freq = 1500
start_date = '2021-01-19'
end_date = '2021-01-20'
#dv1 = np.sin(freq * 2 * np.pi * time) + np.random.normal(0, 0.1, len(time))
#dv2 = np.sin(freq * 2 * np.pi * (time + v12_delay)) + np.random.normal(0, 0.1, len(time))
#dv3 = np.sin(freq * 2 * np.pi * time) + np.random.normal(0, 0.1, len(time))
#dv4 = np.sin(freq * 2 * np.pi * (time + v34_delay)) + np.random.normal(0, 0.1, len(time))

data = wtu.get_data(start_date, end_date)

dv1, dv2, dv3, dv4, __ = data.get_vac_data()

start_date = dv1.columns[0]
time = np.array(dv1.index)
dv1 = np.array(dv1[start_date])
dv2 = np.array(dv2[start_date])
dv3 = np.array(dv3[start_date])
dv4 = np.array(dv4[start_date])

spec, __, xf, __ = wtu.sliding_fft(dv1, time)
plt.plot(xf, spec[0])
plt.show()

single_freq = wtu.frequency_filter(dv1, dv2, dv3, dv4, time, window_size=5000)


print(np.array(single_freq))
