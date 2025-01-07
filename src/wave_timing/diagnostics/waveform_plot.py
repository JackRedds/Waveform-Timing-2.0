import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from wave_timing.math import trig, geometry, analysis


def wave_sliding_fft(wave1, time, ax, wave_norm=True):
    spec1, time_spec, xf, __ = trig.sliding_fft(wave1, time)

    if wave_norm:
        wave1 = analysis.wave_normalization(wave1)

    ax[0].pcolormesh(time_spec, xf, spec1.T, shading='gouraud',
                     cmap='jet', norm=mpl.colors.LogNorm())
    ax[0].set_ylim(0, 1000)
    ax[1].plot(time, wave1, 'r')

def all_waves_sliding_fft(wave1, wave2, wave3, wave4, time, wave_norm=True):
    fig, ax = plt.subplots(8, sharex=True, figsize=(10, 20))
    wave_sliding_fft(wave1, time, ax=ax[0:1], wave_norm=wave_norm)
    wave_sliding_fft(wave2, time, ax=ax[2:3], wave_norm=wave_norm)
    wave_sliding_fft(wave3, time, ax=ax[4:5], wave_norm=wave_norm)
    wave_sliding_fft(wave4, time, ax=ax[6:7], wave_norm=wave_norm)
    ax[0].set_ylabel('$f$ [Hz]')
    ax[1].set_ylabel('dV1')
    ax[2].set_ylabel('$f$ [Hz]')
    ax[3].set_ylabel('dV2')
    ax[4].set_ylabel('$f$ [Hz]')
    ax[5].set_ylabel('dV3')
    ax[6].set_ylabel('$f$ [Hz]')
    ax[7].set_ylabel('dV4')
    ax[7].set_xlabel('time [s]')
    plt.tight_layout()
