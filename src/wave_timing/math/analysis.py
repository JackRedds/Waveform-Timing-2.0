import numpy as np
from astropy import units as u
from wave_timing.math import geometry


def derivitive(sig: np.ndarray, time: np.ndarray):
    assert len(sig) != 0
    # sig must be numpy array
    length = len(sig)
    derivitive = np.zeros(length)
    dt = time[1] - time[0]

    for i in range(length - 1):
        derivitive[i] = (sig[i + 1] - sig[i]) / dt

    return derivitive


def wave_velocity(delay_12, delay_34, boom_length=3.5*u.m):
    assert boom_length > 0
    delay_vector = np.array([delay_12, delay_34]).T * u.s
    delay_mag = np.sqrt(geometry.dot_product(delay_vector, delay_vector))
    velocity = boom_length / delay_mag

    k_vector = velocity * delay_vector.T / boom_length
    velocity = velocity.to(u.km / u.s)
    velocity_vec = velocity.value * k_vector
    return velocity_vec.T


def wave_normalization(wave: np.ndarray):
    mean = np.mean(wave)
    std = np.std(wave)
    return (wave - mean) / std


def divisible(denom, divid):
    remainder = divid % denom
    mask = remainder < 10
    if np.all(mask):
        return True
    else:
        return False
