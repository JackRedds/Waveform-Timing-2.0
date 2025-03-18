import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from wave_timing.math import trig, geometry, analysis


def plot_waveform(time, axes, wave_norm=True, **waves):
    colors = ['k', 'r']
    for label, wave, color in zip(waves.keys(), waves.values(), colors):
        if wave_norm:
            wave = analysis.wave_normalization(wave)

        axes.plot(time, wave, label=label, color=color)
    axes.legend(loc = 'upper right')

def plot_sliding_fft(time, axes, wave_norm=True, **waves):
    for ax, label, wave in zip(axes, waves.keys(), waves.values()):
        if wave_norm:
            wave = analysis.wave_normalization(wave)
        spec, time_spec, xf, __ = trig.sliding_fft(wave, time)

        ax.pcolormesh(time_spec, xf, spec.T, shading='gouraud',
                     cmap='jet', norm=mpl.colors.LogNorm())
        ax.set_ylim(0, 1000)
        ax.set_ylabel(label)

    ax.set_xlabel('Time (s)')
    plt.tight_layout()
