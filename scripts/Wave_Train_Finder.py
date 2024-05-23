#IMPORTS
import cdflib
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
import csv
from astropy import constants as const
import statistics
import os
from scipy.signal import firwin, freqz
from astropy import units as u
import matplotlib as mpl
from scipy.fft import fft, fftfreq

# Code parameters:
#################################################################################################################################

# Change these to switch the datafile being run
#------------------------------------------------------------------------------------
date_wave = ''
while len(date_wave) != 10:
  date_wave = input('What File would you like to Run: ')
  if len(date_wave) != 10:
    print("Enter Valid Time")

# Bandpass Filter Parameters
#------------------------------------------------------------------------------------
# Low Cut Frequency (rec:100; Moz: 500)
low_cut_freq = 100
# High Cut Frequency (rec:900; Moz: 1100)
high_cut_freq = 900
# Filter length (rec: 601)
filt_len = 601

# Set frequency and potential type and plots
#------------------------------------------------------------------------------------
# which type of filter (Fir for fir filter; rec: Fir)
filt_type = 'Fir'
# which type of potential subtraction ('Cat' for cattell, 'Moz' for Mozer, 'None' for none)
pot_type = 'Moz'
# Plot FFTs and polar hist
plot_stuff = False
# vdc or vac?
file_type = 'vac'
# Include V5?
v5_inclde = True

# Parameters for frequency filter
#------------------------------------------------------------------------------------
# window size (rec: 5000; 500)
window_size_ff = 5000
# hop size (rec: 2500; 250)
hop_size_ff = 2500

# Parameters for cross correlation
#------------------------------------------------------------------------------------
# window size (rec: 500)
window_size_x_corr = 5000

# Constants
#------------------------------------------------------------------------------------
#Boom length is about ~3.5m
boom_len = 3.5

# Which Trial would you like to start with?
#------------------------------------------------------------------------------------
start_trial = 0

#################################################################################################################################

dpath = os.path.dirname(os.path.realpath(__file__))
wpath = os.path.join(dpath, 'data_output')
if os.path.exists(wpath) == False:
    os.mkdir(wpath)
wpath = os.path.join(wpath, 'Date {}'.format(date_wave))
figpath = os.path.join(wpath, 'Full Figure')
figpathx = os.path.join(wpath, 'X_corr Figures')
# dpath = os.path.join(dpath, 'Data')
parent = os.path.dirname(dpath)
fld_vac = os.path.join(parent, "psp_data","fields","l2",f"dfb_dbm_{file_type}",f"{date_wave[0:4]}")
fld_mag = os.path.join(parent, "psp_data","fields","l2","mag_sc_4_per_cycle",f"{date_wave[0:4]}")
swp_sw = os.path.join(parent, "psp_data","sweap","spi","l3","spi_sf00_l3_mom",f"{date_wave[0:4]}")
if os.path.exists(wpath) == False:
    os.mkdir(wpath)
if os.path.exists(figpath) == False:
    os.mkdir(figpath)
if os.path.exists(figpathx) == False:
    os.mkdir(figpathx)

# Function Calls:
#################################################################################################################################
def find_nearest(array, value):
    """
    Find the nearest value in the array to the value given

    :param array: array of numbers
    :param value: 
    :return: The index of the array with the number closest to value given
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    #idx = (array - value).argmin()
    return idx

def x_corr(array1, array2, start, end, time, date, array1_str='dV', array2_str='dV'):
    """
    Calculating the cross correlation between two arrays

    :param array1: First array being correlated
    :param array2: Second array being correlated
    :param start: Index in array where cross correlation begins
    :param end: Index in the array where cross correlation ends
    :param array1_str: String discribing the first array
    :param array2_str: String discribing the second array
    :param time: Time related to the arrays
    :param date: String of the date
    :return: The time delay attained from the cross correlation
    """

    norm1 = (array1[start:end] - np.mean(array1[start:end])) / (np.std(array1[start:end]) * len(array1[start:end]))
    norm2 = (array2[start:end] - np.mean(array2[start:end])) / (np.std(array2[start:end]))
    correlation = signal.correlate(norm1, norm2, mode="full", method = "fft")
    lags = signal.correlation_lags(len(norm1),len(norm2), mode = "full")
    lags = lags*time[0][1]
    xmax = lags[np.argmax(correlation)]
    # plt.plot(lags, correlation)
    # # plt.title(date[0:11] + " " + array1_str + " vs " + array2_str + " Corrilation")
    # plt.xlabel('Time Lag')
    # plt.ylabel("Correlation(seconds)")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(time[0,start:end], array1[start:end], label='dv1')
    # ax.plot(time[0,start:end], array2[start:end], label='dv2')
    # ax.plot(time[0,start:end]+xmax, array2[start:end], label=f'dv2 shifted corr: {round(correlation.max(),2)}')
    # ax.set_xlabel('time [s]')
    # ax.legend()
    # # fig.savefig(f"./Summer_Update/correlation-Trial{i}-time-{round(time[0,start],2)}-{round(time[0,end],2)}.png")
    # plt.show()
    return xmax, correlation

def angle_non_orth(in_angle, t_12, t_34):
    """
    Using the initial angle attained by assuming the angle between V1-2 and V3-4 is 90 degrees to find the actual angle
    between V1-2 and V3-4 within a degree

    :param in_angle: Initial angle attained from assuming the angle between V1-2 and V3-4 is 90 degrees
    :param t_12: Time lag between V1-2
    :param t_34: Time lag between V3-4
    :return: The new angle attained when V1-2 and V3-4 are non-orthogonal
    """

    angle = in_angle
    if t_12 == 0:
        if t_34 > 0:
            angle = 90
        else:
            angle = 270
    else:
        frac = t_34/t_12
        minum = abs((np.sin((in_angle-5-15)*(math.pi/180))/np.cos((in_angle-15)*math.pi/180))-frac)
        for i in range(1, 30):
            diff = abs((np.sin((in_angle-5-15+i)*(math.pi/180))/np.cos((in_angle-15+i)*math.pi/180))-frac)
            if diff < minum:
                angle = in_angle-15+i
    return angle

def debye_length(elec_temp_par, elec_temp_per, elec_density):
    #Never Fully Implemented
    """
    Calculating the debye length

    :param elec_temp_par: Parrellel component of the electron temperature (eV)
    :param elec_temp_per: Perpindicular component of the electron temperature (eV)
    :param elec_density: electron density (cm^-3)
    :const const.e: electron charge
    :const const.eps0: epsilon nought or the permitivity constant
    :return: The debye length
    """
    elec_density_m3 = elec_density*1000000
    elec_temp = np.sqrt((elec_temp_par**2)+(elec_temp_per**2))
    debye_len = np.sqrt((const.eps0*elec_temp)/(elec_density_m3*const.e))
    return debye_len

def bandpass(wa, wb, n):
    """
    Ideal bandpass FIR filter.

    Frequencies normalized to the Nyquist frequency. Can be [0, 1]

    :param wa: Low cutoff frequency (Hz)
    :param wb: High cutoff frequency (Hz)
    :param n: The filter length. Is odd n = 2*m + 1
    :return: The filter array of length n
    """
    wa = wa * np.pi
    wb = wb * np.pi
    m = (n - 1) // 2
    k = np.arange(-m, m + 1)
    k[m] = 1.0
    d = (np.sin(wb * k) / (np.pi * k)) - (np.sin(wa * k) / (np.pi * k))
    d[m] = (wb - wa) / np.pi
    return d

def filter(low_cut, high_cut, width, wave):
    """
    Filters the electric wave

    :param low_cut: Low cutoff frequency (Hz)
    :param high_cut: High cutoff frequency (Hz)
    :param width: The filter length. Is odd n = 2*m + 1
    :param wave: Unfiltered electric wave
    :return: Filtered electric wave
    """
    # Note: Wider filters (higher width) can remove lower frequencies.
    # Width must be odd.

    # Normalized frequencies.
    w_low = low_cut / nyquist
    w_high = high_cut / nyquist

    # Make an ideal band pass filter.
    h = bandpass(w_low, w_high, width)
    # Apply Hamming window to remove ripples.
    #h *= hamming(width)
    y = np.convolve(h, wave, 'same')

    #fig, ax = plt.subplots()
    #ax.plot(time[time_sample_index], wave, '.-', label='Non-Filtered')
    #ax.plot(time[time_sample_index], y, '.-', label='Filtered')
    #ax.grid(True)
    #ax.set_xlabel('Time (s)')
    #ax.legend()
    #plt.show()

    return y

def find_phase(corr, fs, peak_width = 5):

    """
    Finds the phase, frequency, and time delay using the cross corrilation

    :param corr: Cross corrrilation of the two waves
    :param fs: Sample rate
    :param peak_width: Width of the peak which is set to 11
    :return: The phase(deg), frequency(Hz), and delay(sec) of the two waves
    """
    corr_max = signal.find_peaks(corr, width=peak_width)[0]
    # plt.plot(corr)
    # plt.plot(corr_max, corr[corr_max],'x')
    # plt.show()
    if len(corr_max) > 5:
        n_per_cycle = np.mean(np.diff(corr_max)[2:-2])
    else:
        n_per_cycle = np.mean(np.diff(corr_max))
    period = n_per_cycle/fs
    frequency = 1.0/period
    shifts = corr_max - (len(corr) // 2)
    i_delay = np.argmin(np.abs(shifts))
    fract = shifts[i_delay] / n_per_cycle
    phase = fract * 360.0
    delay = fract * period
    return phase, frequency, delay

def n_x_corr(array1, array2, num_x_corr, time, arr_str_1, arr_str_2):
    """
    Breaks wave up into multiple sections and finds the corrilation of each one

    :param array1: 
    :param array2: 
    :param num_x_corr: 
    :param time:
    :param arr_str_1
    :param arr_str_2
    :return: 
    """
    arr_len = len(array1)
    par_array = arr_len//num_x_corr
    x_corr_lst = [0]*num_x_corr
    x_corr_2 = [0]*num_x_corr
    start = 0
    end = par_array
    # corr = []
    for i in range(0,num_x_corr):
        # fig, ax = plt.subplots()
        x_corr_lst[i] = x_corr(array1, array2, start, end, time, date, arr_str_1, arr_str_2)
        # x_corr_2[i] = find_phase(x_corr_lst[i][1], samp_rt)[2]
        # time_lst = time[1] + x_corr_lst[i][0]
        # time_2 = time[1] + x_corr_2[i][2]
        # ax.plot(time[1][start:end],array1[start:end], label = arr_str_1)
        # ax.plot(time[1][start:end],array2[start:end], label = arr_str_2)
        # ax.plot(time_lst[start:end], array2[start:end], label = arr_str_2 + " original")
        # ax.plot(time_2[start:end], array2[start:end], label = arr_str_2 + " mike")
        # plt.legend()
        # plt.show()
        start = start + par_array
        end = end + par_array
        # corr.append(x_corr_lst[1])
    
    # xf = np.abs(fftfreq(window_size_x_corr, 1 / samp_rt))
    # time_bins = np.linspace(0, 3.5, window_size_x_corr)

    # fig,ax = plt.subplots()
    # im = ax.pcolormesh(time_bins, xf, corr.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    # cb = fig.colorbar(im)
    # cb.set_label("dV1", size="large")
    # # ax.set_ylim(0,2000)
    # ax.set_ylabel('Frequency [Hz]')

    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].plot(np.linspace(0, time[0,len(time[0])-1],num_x_corr), x_corr_2, '.')
    # ax[1].plot(time[0], array1)
    # ax[1].plot(time[0], array2)
    # plt.show()
    return np.linspace(0, time[0,len(time[0])-1],num_x_corr), x_corr_lst

def outliers(arr):
    #Not Used in This Version of the code
    """
    Gets rid of all outliers in the array of cross corrilations

    :param arr: Array of cross corrilations
    :return: The average of the cross corrilations with the outliers removed
    """
    arr = np.array(arr)
    std = statistics.stdev(arr)
    valid = (np.abs(arr) < 2 * std)
    arr = arr[valid]

    #if not np.sum(valid):
     #   print("NOT VALID!!! You DID SOMETHING WWRROONNGG!!!!")
      #  exit()
    #else:
    return np.mean(arr)

def find_peaks(arr1, arr2):
    """
    Finds the peaks and troughs of the two input arrays

    :param arr1: array of numbers
    :param arr2: array of numbers
    :return: The peaks and troughs of the two arrays
    """

    peaks1, _ = signal.find_peaks(arr1,distance = 50, prominence = 0.5*np.mean(arr1))
    troughs1,_ = signal.find_peaks(-arr1, distance = 50, prominence = 0.5*np.mean(arr1))
    peaks2,_ = signal.find_peaks(arr2,distance = 50, prominence = 0.5*np.mean(arr2))
    troughs2,_ = signal.find_peaks(-arr2, distance = 50, prominence = 0.5*np.mean(arr2))
    
    # fig,ax = plt.subplots()
    # ax.plot(arr1)
    # ax.plot(peaks1, arr1[peaks1], 'rx')
    # ax.plot(troughs1, arr1[troughs1], 'kx')
    # ax.plot(arr2)
    # ax.plot(peaks2, arr2[peaks2], 'rx')
    # ax.plot(troughs2, arr2[troughs2], 'kx')
    
    # plt.show()

    return peaks1, troughs1, peaks2, troughs2

def deriv(arr1, arr2, time):
    """
    Takes numerical derivitive of two inputted arrays

    :param arr1: First array
    :param arr2: Second array
    :param time: Array of time coresponding to the arrays
    :return: The dervivitives of the two arrays
    """
    deriv_arr1 = np.zeros(len(arr1))
    deriv_arr2 = np.zeros(len(arr2))
    td = time[0,1]-time[0,0]
    for i in range(len(arr1)-1):
        deriv_arr1[i] = (arr1[i+1]-arr1[i])
        deriv_arr2[i] = (arr2[i+1]-arr2[i])
    return deriv_arr1, deriv_arr2

def trim_v3(arr1, arr2):

    """
    Trim the longer array of the two down to the length of the shorter one by finding the closest
    values of the longer array to the shorter array

    :param arr1: array of numbers
    :param arr2: array of numbers
    :return: The two arrays, now of the same length
    """
    if len(arr2)>len(arr1):
        arr2_new = np.zeros(len(arr1))
        for i in range(len(arr1)):
            idx = find_nearest(arr2,arr1[i])
            arr2_new[i] = arr2[idx]
        arr2 = arr2_new
    elif len(arr2)<len(arr1):
        arr1_new = np.zeros(len(arr2))
        for i in range(len(arr2)):
            idx = find_nearest(arr1, arr2[i])
            arr1_new[i] = arr1[idx]
        arr1 = arr1_new
    return arr1, arr2
class FIRBandPass(object):

    def __init__(self, taps, cutoff, fs):
        """
        A FIR band pass filter.

        :param taps: The length of the filter.
        :param cutoff: Cutoff frequency [low_cut, high_cut]. (Hz)
        :param fs: The signal sample rate (Sa/s)
        """
        self.taps = taps
        self.cutoff = cutoff
        self.fs = fs

        # Make the FIR filter.
        self.nyq = fs / 2.0
        self.taps = firwin(numtaps=taps,
                           cutoff=[low_cut_freq, high_cut_freq],
                           fs=fs,
                           pass_zero=False)

    def filter(self, v, trim=False):
        """
        Filter the given signal.

        :param v: The signal to filter.
        :param trim: If True, use the value nan at the beginning and ending of
            the filtered signal where boundary effects are visible.
        :return: The filtered signal.
        """
        v_filtered = np.convolve(self.taps, v, mode='same')
        if trim:
            v_filtered[:self.taps // 2] = np.nan
            v_filtered[-self.taps // 2:] = np.nan
        return v_filtered

    def plot_gain(self):
        """
        Plot the frequency response of the filter.

        :return: A tuple (fig, ax) where fig is the figure and ax is the axis
            the plot is on.
        """
        w, h = freqz(self.taps, 1.0)

        fig, ax = plt.subplots()
        ax.plot(w / max(w) * self.nyq, abs(h))
        ax.set_ylabel('Gain')
        ax.set_xlabel(r'Freq (Hz)')
        ax.set_title(r'Frequency response')

        return fig, ax

def filter_fir(low_cut, high_cut, width, wave):
    """
    Filters the electric wave using FIRBandPass filter class

    :param low_cut: Low cutoff frequency (Hz)
    :param high_cut: High cutoff frequency (Hz)
    :param width: The filter length. Is odd n = 2*m + 1
    :param wave: Unfiltered electric wave
    :return: Filtered electric wave
    """
    band = FIRBandPass(width, [low_cut, high_cut], fs=nyquist*2)
    filt = band.filter(wave)
    return filt

def frequency(arr):
    """
    Finds the frequency of the wave from an array of points of interest (POI)
    such as peaks, troughs, zero-risings, or zero-fallings

    :param arr: arr of POI
    :return: Calculated frequency
    """
    diff = []
    for i in range(len(arr)-1):
        temp = arr[i+1] - arr[i]
        diff.append(temp)
    dt = time[0][1]-time[0][0]
    diff = np.array(diff)*dt
    freq = 1/diff
    return freq

def rmv_dust_peaks(v1, v2, v3, v4, epoch, time, dust_epoch):
# This is wrong, i needs to be the index that the dust peak occurs at.
    del_rng = 1000
    v1 = v1.tolist()
    v2 = v2.tolist()
    v3 = v3.tolist()
    v4 = v4.tolist()
    time = time.tolist()

    for i, x in enumerate(dust_epoch):
        if epoch[0] <= x <= epoch[len(epoch)-1]:
            t = find_nearest(epoch, x)
            # del v1[t-del_rng:t+del_rng]
            # del v2[t-del_rng:t+del_rng]
            # del v3[t-del_rng:t+del_rng]
            # del v4[t-del_rng:t+del_rng]
            # del time[t-del_rng:t+del_rng]
            print(epoch[t-1])
            print(epoch[t])
            print(x)
            print(dust_epoch[i+1])
            plt.plot(v1)
            plt.show()
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    v4 = np.array(v4)
    time = np.array(time)
    return v1, v2, v3, v4, time

def peak_to_peak(arr, peak, trough):
    """
    Calculates peak to peak of wave

    :param arr: Wave that finding peak to peak of
    :param peak: location of peaks
    :param trough: location of troughs
    :return: array of peak to peak values vs time
    """
    peak = np.array(peak)
    trough = np.array(trough)
    ave_arr = np.zeros(len(peak))
    for i in range(len(peak)):
        trh_less = trough[trough<peak[i]]
        if(len(trh_less)>0):
            idx_less = find_nearest(trh_less, peak[i])
            diff_less = abs(arr[int(trh_less[idx_less])])+abs(arr[int(peak[i])])
        trh_great = trough[trough>peak[i]]
        if(len(trh_great)>0):
            idx_great = find_nearest(trh_great, peak[i])
            diff_great = abs(arr[int(trh_great[idx_great])])+abs(arr[int(peak[i])])
        if(len(trh_less)==0 and len(trh_great)==0):
            break
        elif(len(trh_less)>0 and len(trh_great)>0):
            ave_pk = (diff_great+diff_less)/2
        elif(len(trh_great)==0):
            ave_pk = diff_less
        elif(len(trh_less)==0):
            ave_pk = diff_great
        ave_arr[i] = ave_pk
    return ave_arr

def diff(arr1, arr2):
    """
    Find the delay between the two waves by finding the closest POI of one wave to the other and
    finding the difference between those points

    :param arr1: Array of POI from wave 1
    :param arr2: Array of POI from wave 2
    :return: An array of delays and position of them
    """
    if len(arr2)>len(arr1):
        difference = np.zeros(len(arr1))
        ave_pos = np.zeros(len(arr1))
        for i in range(len(arr1)):
            idx = find_nearest(arr2,arr1[i])
            difference[i] = arr2[idx]-arr1[i]
            ave_pos[i] = (arr2[idx]+arr1[i])/2
    elif len(arr2)<=len(arr1):
        difference = np.zeros(len(arr2))
        ave_pos = np.zeros(len(arr2))
        for i in range(len(arr2)):
            idx = find_nearest(arr1, arr2[i])
            difference[i] = arr2[i]-arr1[idx]
            ave_pos[i] = (arr2[i]+arr1[idx])/2

    return difference, ave_pos

def trim_diff(diff1, time1, diff2, time2):
    """
    Trim the arrays of delay to be the same lenth

    :param arr1: array of delays between two waves
    :param arr2: array of delays between two waves
    :return: The two arrays, now of the same length
    """
    if len(time2)>len(time1):
        diff2_new = np.zeros(len(diff1))
        time2_new = np.zeros(len(diff1))
        for i in range(len(time1)):
            idx = find_nearest(time2,time1[i])
            diff2_new[i] = diff2[idx]
            time2_new[i] = time2[idx]
        diff2 = diff2_new
        time2 = time2_new
    elif len(time2)<len(time1):
        diff1_new = np.zeros(len(diff2))
        time1_new = np.zeros(len(diff2))
        for i in range(len(time2)):
            idx = find_nearest(time1, time2[i])
            diff1_new[i] = diff1[idx]
            time1_new[i] = time1[idx]
        diff1 = diff1_new
        time1 = time1_new
    return diff1, time1, diff2, time2

def sliding_fft(sig1, sig2, sig3, sig4, sig5, sig6, window_size, hop_size):
    """
    Compute the sliding FFT of each of the four waves, also save sections of wave
    where only 1 strong frequency is present in all four waves

    :param sig1: Wave 1
    :param sig2: Wave 2
    :param sig3: Wave 3
    :param sig4: Wave 4
    :param window_size: Side of the window being FFT over
    :param hop_size: Size of the shift of the section
    :return: The Spectra of the 4 waves and the sections where a single frequancy in present
    """
    pks_v1, trh_v1, pks_v2, trh_v2 = find_peaks(sig1, sig2)

    pktopk_pks_v1 = peak_to_peak(sig1, pks_v1, trh_v1)   
    pktopk_trh_v1 = peak_to_peak(sig1, trh_v1, pks_v1) 
    v1_mean = (np.mean(pktopk_pks_v1)+np.mean(pktopk_trh_v1))/2
    sig1 = sig1/v1_mean

    pktopk_pks_v2 = peak_to_peak(sig2, pks_v2, trh_v2)   
    pktopk_trh_v2 = peak_to_peak(sig2, trh_v2, pks_v2)
    v2_mean = (np.mean(pktopk_pks_v2)+np.mean(pktopk_trh_v2))/2
    sig2 = sig2/v2_mean

    pks_v3, trh_v3, pks_v4, trh_v4 = find_peaks(sig3, sig4)

    pktopk_pks_v3 = peak_to_peak(sig3, pks_v3, trh_v3)   
    pktopk_trh_v3 = peak_to_peak(sig3, trh_v3, pks_v3) 
    v3_mean = (np.mean(pktopk_pks_v3)+np.mean(pktopk_trh_v3))/2
    sig3 = sig3/v3_mean

    pktopk_pks_v4 = peak_to_peak(sig4, pks_v4, trh_v4)   
    pktopk_trh_v4 = peak_to_peak(sig4, trh_v4, pks_v4)
    v4_mean = (np.mean(pktopk_pks_v4)+np.mean(pktopk_trh_v4))/2
    sig4 = sig4/v4_mean

    pks_v5, trh_v5, pks_v6, trh_v6 = find_peaks(sig5, sig6)

    pktopk_pks_v5 = peak_to_peak(sig5, pks_v5, trh_v5)   
    pktopk_trh_v5 = peak_to_peak(sig5, trh_v5, pks_v5) 
    v5_mean = (np.mean(pktopk_pks_v5)+np.mean(pktopk_trh_v5))/2
    sig5 = sig5/v5_mean

    pktopk_pks_v6 = peak_to_peak(sig6, pks_v6, trh_v6)   
    pktopk_trh_v6 = peak_to_peak(sig6, trh_v6, pks_v6)
    v6_mean = (np.mean(pktopk_pks_v6)+np.mean(pktopk_trh_v6))/2
    sig6 = sig6/v6_mean


    signal_length = len(sig1)
    num_windows = (signal_length - window_size) // hop_size + 1
    spectra1 = []
    spectra2 = []
    spectra3 = []
    spectra4 = []
    spectra5 = []
    spectra6 = []
    single_freq = []
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window1 = sig1[start:end] * np.hanning(window_size)  # Apply a Hanning window
        spectrum1 = np.abs(fft(window1)/window_size)  # Compute FFT
        window2 = sig2[start:end] * np.hanning(window_size)  # Apply a Hanning window
        spectrum2 = np.abs(fft(window2)/window_size)
        window3 = sig3[start:end] * np.hanning(window_size)  # Apply a Hanning window
        spectrum3 = np.abs(fft(window3)/window_size)
        window4 = sig4[start:end] * np.hanning(window_size)  # Apply a Hanning window
        spectrum4 = np.abs(fft(window4)/window_size)
        window5 = sig5[start:end] * np.hanning(window_size)  # Apply a Hanning window
        spectrum5 = np.abs(fft(window5)/window_size)
        window6 = sig6[start:end] * np.hanning(window_size)  # Apply a Hanning window
        spectrum6 = np.abs(fft(window6)/window_size)
        N = window_size
        xf = fftfreq(N, 1 / samp_rt)

        peaks1, _ = signal.find_peaks(spectrum1[0:window_size//5], prominence = 0.2)
        peaks2, _ = signal.find_peaks(spectrum2[0:window_size//5], prominence = 0.2)
        peaks3, _ = signal.find_peaks(spectrum3[0:window_size//5], prominence = 0.2)
        peaks4, _ = signal.find_peaks(spectrum4[0:window_size//5], prominence = 0.2)
        peaks5, _ = signal.find_peaks(spectrum5[0:window_size//5], prominence = 0.2)
        peaks6, _ = signal.find_peaks(spectrum6[0:window_size//5], prominence = 0.2)

        small_peaks1, _ = signal.find_peaks(spectrum1[0:window_size//5], prominence = [0.1,0.2])
        small_peaks2, _ = signal.find_peaks(spectrum2[0:window_size//5], prominence = [0.1,0.2])
        small_peaks3, _ = signal.find_peaks(spectrum3[0:window_size//5], prominence = [0.1,0.2])
        small_peaks4, _ = signal.find_peaks(spectrum4[0:window_size//5], prominence = [0.1,0.2])
        small_peaks5, _ = signal.find_peaks(spectrum5[0:window_size//5], prominence = [0.1,0.2])
        small_peaks6, _ = signal.find_peaks(spectrum6[0:window_size//5], prominence = [0.1,0.2])

        # peaks1, _ = signal.find_peaks(spectrum1[0:1000], height = 1)
        # peaks2, _ = signal.find_peaks(spectrum2[0:1000], height = 1)
        # peaks3, _ = signal.find_peaks(spectrum3[0:1000], height = 1)
        # peaks4, _ = signal.find_peaks(spectrum4[0:1000], height = 1)

        # small_peaks1, _ = signal.find_peaks(spectrum1[0:1000], height = [0.5,1])
        # small_peaks2, _ = signal.find_peaks(spectrum2[0:1000], height = [0.5,1])
        # small_peaks3, _ = signal.find_peaks(spectrum3[0:1000], height = [0.5,1])
        # small_peaks4, _ = signal.find_peaks(spectrum4[0:1000], height = [0.5,1])

        # print(f"value to get 1: {1/v1_mean}")
        # print(f"value to get 0.5: {0.5/v1_mean}")

        # fig, ax = plt.subplots(4,figsize=(12, 6),sharex=True)
        # ax[0].axhline(y = 0.1, color = 'k')
        # ax[2].axhline(y = 0.1, color = 'k')
        # ax[0].axhline(y = 0.05, color = 'k')
        # ax[2].axhline(y = 0.05, color = 'k')
        # ax[0].plot(np.abs(xf), spectrum1,'b')
        # ax[0].plot(np.abs(xf), spectrum2,'g')
        # ax[2].plot(np.abs(xf), spectrum3,'r')
        # ax[2].plot(np.abs(xf), spectrum4,'y')
        # ax[0].plot(xf[peaks1],spectrum1[peaks1],'bx')
        # ax[0].plot(xf[peaks2],spectrum2[peaks2],'gx')
        # ax[2].plot(xf[peaks3],spectrum3[peaks3],'rx')
        # ax[2].plot(xf[peaks4],spectrum4[peaks4],'yx')

        # ax[0].plot(xf[small_peaks1],spectrum1[small_peaks1],'bx')
        # ax[0].plot(xf[small_peaks2],spectrum2[small_peaks2],'gx')
        # ax[2].plot(xf[small_peaks3],spectrum3[small_peaks3],'rx')
        # ax[2].plot(xf[small_peaks4],spectrum4[small_peaks4],'yx')
        # ax[0].set_ylabel('Power')
        # ax[2].set_ylabel('Power')
        # ax[0].set_xlim(0,5000)
        # ax[2].set_xlim(0,5000)
        # ax[1].plot(sig1[start:end],'b', label = 'dv1')
        # ax[1].plot(sig2[start:end],'g',label='dv2')
        # ax[3].plot(sig3[start:end],'r',label='dv3')
        # ax[3].plot(sig4[start:end],'y',label='dv4')
        # ax[1].legend()
        # ax[3].legend()
        # fig.tight_layout(w_pad=0.05, h_pad=0.05)
        # # fig.savefig(f"./Summer_Update/FFT-time-since-{start}-Trial{i}")
        # plt.show()

        if len(peaks1) == 1 and len(peaks2) == 1 and len(peaks3) == 1 and len(peaks4) == 1 and len(peaks5) == 1 and len(peaks6) == 1:
            if len(small_peaks1) == 0 and len(small_peaks2) == 0 and len(small_peaks3) == 0 and len(small_peaks4) == 0 and len(small_peaks5) == 0 and len(small_peaks6) == 0:
                single_freq.append([start, end])

        # if len(peaks1) == 1 and len(peaks2) == 1:
        #     if len(small_peaks1) == 0 and len(small_peaks2) == 0:
        #         single_freq.append([start, end])

        # print(single_freq)

        spectra1.append(spectrum1)
        spectra2.append(spectrum2)
        spectra3.append(spectrum3)
        spectra4.append(spectrum4)
        spectra5.append(spectrum5)
        spectra6.append(spectrum6)
        # print(single_freq)
    return np.array(spectra1), np.array(spectra2), np.array(spectra3), np.array(spectra4), np.array(spectra5), np.array(spectra6), np.array(single_freq)

def polar_2d_hist(wave_angle, wave_magtd, mag_angle, mag_magtd, wind_angle, wind_magtd):
    """
    Plot a 2d histogram on a polar plot to show the statistics of the wave velocity and angle as well as 
    magnetic field and solar wind angle and magnitude

    :param wave_angle: array of wave angles wrt V1 boom
    :param wave_magtd: array of wave velocities
    :param mag_angle: angle of magnetic field wrt V1 boom
    :param mag_magtd: magnitude of magnetic field in x-y plane
    :param wind_angle: angle of solar wind wrt V1 boom
    :param wind_magtd: magnitude of solar wind in x-y plane
    """
    rbins = np.linspace(0, wave_magtd.max(),5)
    abins = np.linspace(0, 2*np.pi, 30)

    #calculate histogram
    hist, _, _ = np.histogram2d(wave_angle, wave_magtd, bins=(abins, rbins))
    hist[hist<50] = 0
    A, R = np.meshgrid(abins, rbins)

    # plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    ax.vlines(0, 0, wave_magtd.max(), 'k',linewidth=3)
    ax.text(0,wave_magtd.max(),'V1', color = 'r')
    ax.vlines(np.pi,0, wave_magtd.max(), 'k',linewidth=3)
    ax.text(np.pi,wave_magtd.max(),'V2', color = 'r')
    ax.vlines(85*(np.pi/180),0, wave_magtd.max(), 'k',linewidth=3)
    ax.text(85*(np.pi/180),wave_magtd.max(),'V3', color = 'r')
    ax.vlines(265*(np.pi/180),0, wave_magtd.max(), 'k',linewidth=3)
    ax.text(265*(np.pi/180),wave_magtd.max(),'V4', color = 'r')
    ax.vlines(mag_angle, 0, mag_magtd, 'b', linewidth = 3, label=f'Magnetic Field {round(mag_magtd,2)} [nT]')
    ax.vlines(wind_angle, 0, wind_magtd,'r', linewidth=3,label=f'Solar Wind {round(wind_magtd,2)} [km/s]')
    pc = ax.pcolormesh(A, R, hist.T, cmap = 'inferno', norm=mpl.colors.LogNorm())
    # pc = ax.pcolormesh(A, R, hist.T, cmap = 'magma_r', vmin=10, vmax=hist.max())
    # pc = ax.pcolormesh(A, R, hist.T,cmap="magma_r", vmin=hist.min(), vmax=hist.max())
    ax.set_xlabel('Angle [Deg]')
    ax.set_ylabel('Speed [km/s]')
    plt.legend()
    cb = fig.colorbar(pc)
    cb.set_label("Wave", size="large")
    ax.grid(True)
    plt.show()

def angle_wrt_v1(magx, magy, wave_angle=False):
    """
    Calculate the angle wrt V1 boom

    :param magx: magnitude in x direction
    :param magy: magnitude in y direction
    :param wave_angle: boolean to tell function if calculation of of the wave angle
    :return: angle wrt the V1 boom
    """
    if wave_angle == True:
        if t_12 == 0:
            theta = math.pi/2
        else:
            theta = np.arctan(t_34/t_12)
        theta = angle_non_orth(theta*(180/math.pi),t_12,t_34)
    else:
        if magx==0:
            if magy>0:
                theta = 90
            else:
                theta=270
        theta = np.arctan(magy/magx)*(180/np.pi)

    if magx>0 and magy>=0:
        theta = theta
    elif magx<0:
        theta = 180 + theta
    elif magx>0 and magy<0:
        theta = 360 + theta
    return theta

def frequency_filter(arr1, arr2, arr3, arr4, arr5):
    """
    Find sections of the wave where there is only a single strong frequncy and find the delay between 
    each of the two waves in that section. Go through the whole wave doing this.

    :param arr1: Wave 1
    :param arr2: Wave 2
    :param arr3: Wave 3
    :param arr4: Wave 4
    :return: the delays of the peaks and throughs of the waves and the positions of the delays
    """
    window_size = window_size_ff
    hop_size = hop_size_ff
    if v5_inclde == True:
        sc = (arr1+arr2+arr3+arr4)/4
    else:
        sc = arr2
    _,_,_,_,_,_, times = sliding_fft(arr1,arr2,arr3,arr4,arr5,sc, window_size, hop_size)
    diff_pks_v1_v2_full = []
    pos_pks_v1_v2_full = []
    diff_pks_v3_v4_full = []
    pos_pks_v3_v4_full = []
    diff_pks_v1_v5_full = []
    pos_pks_v1_v5_full = []
    diff_trh_v1_v2_full = []
    pos_trh_v1_v2_full = []
    diff_trh_v3_v4_full = []
    pos_trh_v3_v4_full = []
    diff_trh_v1_v5_full = []
    pos_trh_v1_v5_full = []
    for i in range(len(times)):
        pks_v1, trh_v1, pks_v2, trh_v2 = find_peaks(arr1[times[i,0]:times[i,1]], arr2[times[i,0]:times[i,1]])+times[i,0]
        pks_v3, trh_v3, pks_v4, trh_v4 = find_peaks(arr3[times[i,0]:times[i,1]], arr4[times[i,0]:times[i,1]])+times[i,0]
        pks_v5, trh_v5, pks_sc, trh_sc = find_peaks(arr5[times[i,0]:times[i,1]], sc[times[i,0]:times[i,1]])+times[i,0]
        # pks_v1, trh_v1, pks_v2, trh_v2 = find_peaks(arr1[times[i,0]:times[i,1]], arr2[times[i,0]:times[i,1]])
        # pks_v3, trh_v3, pks_v4, trh_v4 = find_peaks(arr3[times[i,0]:times[i,1]], arr4[times[i,0]:times[i,1]])

        # V1-V2
        diff_pks_v1_v2, pos_pks_v1_v2 = diff(pks_v2, pks_v1)
        diff_trh_v1_v2, pos_trh_v1_v2 = diff(trh_v2, trh_v1)

        # V3-V4
        diff_pks_v3_v4, pos_pks_v3_v4 = diff(pks_v4, pks_v3)
        diff_trh_v3_v4, pos_trh_v3_v4 = diff(trh_v4, trh_v3)

        # V1-V5
        diff_pks_v1_v5, pos_pks_v1_v5 = diff(pks_v1, pks_v5)
        diff_trh_v1_v5, pos_trh_v1_v5 = diff(trh_v1, trh_v5)

        # V2-V5
        diff_pks_v2_v5, pos_pks_v2_v5 = diff(pks_v2, pks_v5)
        diff_trh_v2_v5, pos_trh_v2_v5 = diff(trh_v2, trh_v5)

        diff_pks_v1_v5, pos_pks_v1_v5, diff_pks_v2_v5, pos_pks_v2_v5 = trim_diff(diff_pks_v1_v5, pos_pks_v1_v5, diff_pks_v2_v5, pos_pks_v2_v5)
        diff_trh_v1_v5, pos_trh_v1_v5, diff_trh_v2_v5, pos_trh_v2_v5 = trim_diff(diff_trh_v1_v5, pos_trh_v1_v5, diff_trh_v2_v5, pos_trh_v2_v5)


        diff_pks_vsc_v5 = (diff_pks_v1_v5+diff_pks_v2_v5)/2
        pos_pks_vsc_v5 = (pos_pks_v1_v5+pos_pks_v2_v5)/2
        diff_trh_vsc_v5 = (diff_trh_v1_v5+diff_trh_v2_v5)/2
        pos_trh_vsc_v5 = (pos_trh_v1_v5+pos_trh_v2_v5)/2


        diff_pks_v1_v2, pos_pks_v1_v2, diff_pks_v3_v4, pos_pks_v3_v4 = trim_diff(diff_pks_v1_v2, pos_pks_v1_v2, diff_pks_v3_v4, pos_pks_v3_v4)
        diff_trh_v1_v2, pos_trh_v1_v2, diff_trh_v3_v4, pos_trh_v3_v4 = trim_diff(diff_trh_v1_v2, pos_trh_v1_v2, diff_trh_v3_v4, pos_trh_v3_v4)

        diff_pks_v1_v2, pos_pks_v1_v2, diff_pks_vsc_v5, pos_pks_vsc_v5 = trim_diff(diff_pks_v1_v2, pos_pks_v1_v2, diff_pks_vsc_v5, pos_pks_vsc_v5)
        diff_trh_v1_v2, pos_trh_v1_v2, diff_trh_vsc_v5, pos_trh_vsc_v5 = trim_diff(diff_trh_v1_v2, pos_trh_v1_v2, diff_trh_vsc_v5, pos_trh_vsc_v5)

        diff_pks_vsc_v5, pos_pks_vsc_v5, diff_pks_v3_v4, pos_pks_v3_v4 = trim_diff(diff_pks_vsc_v5, pos_pks_vsc_v5, diff_pks_v3_v4, pos_pks_v3_v4)
        diff_trh_vsc_v5, pos_trh_vsc_v5, diff_trh_v3_v4, pos_trh_v3_v4 = trim_diff(diff_trh_vsc_v5, pos_trh_vsc_v5, diff_trh_v3_v4, pos_trh_v3_v4)
      
        # # plt.plot(pks_v1, arr1[times[i,0]:times[i,1]][pks_v1],'x')
        # plt.plot(np.linspace(times[i,0],times[i,1],times[i,1]-times[i,0]), arr1[times[i,0]:times[i,1]]*10000)
        # # plt.plot(np.linspace(times[i,0],times[i,1],times[i,1]-times[i,0]), arr1[times[i,0]:times[i,1]])
        # # plt.plot(arr1[times[i,0]:times[i,1]])
        # # plt.plot(pks_v2, arr2[times[i,0]:times[i,1]][pks_v2],'x')
        # plt.plot(np.linspace(times[i,0],times[i,1],times[i,1]-times[i,0]),arr2[times[i,0]:times[i,1]]*10000)
        # # plt.plot(np.linspace(times[i,0],times[i,1],times[i,1]-times[i,0]),arr2[times[i,0]:times[i,1]])
        # # plt.plot(arr2[times[i,0]:times[i,1]])
        # plt.plot(pos_pks_v1_v2, diff_pks_v1_v2, '.')
        # plt.plot(pos_trh_v1_v2, diff_trh_v1_v2, '.')
        # plt.show()

        diff_pks_v1_v2_full.extend(diff_pks_v1_v2)
        pos_pks_v1_v2_full.extend(pos_pks_v1_v2)
        diff_pks_v3_v4_full.extend(diff_pks_v3_v4)
        pos_pks_v3_v4_full.extend(pos_pks_v3_v4)
        diff_pks_v1_v5_full.extend(diff_pks_vsc_v5)
        pos_pks_v1_v5_full.extend(pos_pks_vsc_v5)
        diff_trh_v1_v2_full.extend(diff_trh_v1_v2)
        pos_trh_v1_v2_full.extend(pos_trh_v1_v2)
        diff_trh_v3_v4_full.extend(diff_trh_v3_v4)
        pos_trh_v3_v4_full.extend(pos_trh_v3_v4)
        diff_trh_v1_v5_full.extend(diff_trh_vsc_v5)
        pos_trh_v1_v5_full.extend(pos_trh_vsc_v5)

        # plt.plot(pos_pks_v1_v5_full,diff_pks_v1_v5_full)
        # plt.show()
        # print(diff_pks_v1_v5_full)

        # diff_pks_v1_v2_full.append(np.mean(diff_pks_v1_v2))
        # pos_pks_v1_v2_full.append(np.mean(pos_pks_v1_v2))
        # diff_pks_v3_v4_full.append(np.mean(diff_pks_v3_v4))
        # pos_pks_v3_v4_full.append(np.mean(pos_pks_v3_v4))
        # diff_pks_v1_v5_full.append(np.mean(diff_pks_vsc_v5))
        # pos_pks_v1_v5_full.append(np.mean(pos_pks_vsc_v5))
        # diff_trh_v1_v2_full.append(np.mean(diff_trh_v1_v2))
        # pos_trh_v1_v2_full.append(np.mean(pos_trh_v1_v2))
        # diff_trh_v3_v4_full.append(np.mean(diff_trh_v3_v4))
        # pos_trh_v3_v4_full.append(np.mean(pos_trh_v3_v4))
        # diff_trh_v1_v5_full.append(np.mean(diff_trh_vsc_v5))
        # pos_trh_v1_v5_full.append(np.mean(pos_trh_vsc_v5))

    return diff_pks_v1_v2_full, pos_pks_v1_v2_full, diff_pks_v3_v4_full, pos_pks_v3_v4_full,diff_pks_v1_v5_full, pos_pks_v1_v5_full, diff_trh_v1_v2_full, pos_trh_v1_v2_full, diff_trh_v3_v4_full, pos_trh_v3_v4_full,diff_trh_v1_v5_full, pos_trh_v1_v5_full

def frequency_plot(sig1, sig2, start=0, stop=3.5, str1='dV', str2='dV'):
    """
    Plot the sliding FFTs of each of the two waves, along with the two waves, the delays from the peaks and troughs,
    the delay from the sliding X_corr, and the peak to peak vs time of the waves

    :param sig1: Wave 1
    :param sig2: Wave 2
    :param start: start time of plot
    :param stop: end time of plot
    :param str1: string describing sig 1
    :param str2: string describing sig 2
    """
    start = int(start/dt)
    stop = int(stop/dt)
    sig1 = sig1[start:stop]
    sig2 = sig2[start:stop]

    sig1_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, sig1)
    sig2_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, sig2)

    window_size = 5000
    hop_size = 2500
    spectra1, spectra2, _, _, _, _, _ = sliding_fft(sig1,sig2,sig1,sig2,sig1,sig2, window_size, hop_size)

    xf = np.abs(fftfreq(window_size, 1 / samp_rt))
    figure, axes = plt.subplots(6, figsize=(12, 8), sharex=True)
    figure.suptitle("Figure 4", fontsize=25, weight='bold')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    
    time_bins = np.linspace(start*dt, stop*dt, len(spectra1))
    ax = axes[0]
    im = ax.pcolormesh(time_bins, xf, spectra1.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    ax.set_ylim(0,1000)
    ax.set_ylabel(str1 +' [Hz]', fontsize="large")
    ax.grid(False)
    ax.text(0.05, 0.95, 'a', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    

    time_bins = np.linspace(start*dt, stop*dt, len(spectra2))
    ax = axes[1]
    im = ax.pcolormesh(time_bins, xf, spectra2.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    ax.set_ylim(0,1000)
    ax.set_ylabel(str2 + ' [Hz]', fontsize="large")
    ax.grid(False)
    ax.text(0.05, 0.95, 'b', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    pks_v1, trh_v1, pks_v2, trh_v2 = find_peaks(sig1_filt, sig2_filt)

    pktopk_pks_v1 = peak_to_peak(sig1_filt, pks_v1, trh_v1)   
    pktopk_trh_v1 = peak_to_peak(sig1_filt, trh_v1, pks_v1) 

    pktopk_pks_v2 = peak_to_peak(sig2_filt, pks_v2, trh_v2)   
    pktopk_trh_v2 = peak_to_peak(sig2_filt, trh_v2, pks_v2) 

    diff_pks_v1_v2, pos_pks_v1_v2 = diff(pks_v1, pks_v2)
    diff_trh_v1_v2, pos_trh_v1_v2 = diff(trh_v1, trh_v2)

    ax = axes[2]
    ax.plot(np.linspace(0, len(sig1_filt), len(sig1_filt))*dt, sig1_filt*1e3,'k',linewidth = '0.5', label= str1)
    ax.plot(np.linspace(0, len(sig2_filt), len(sig2_filt))*dt, sig2_filt*1e3, 'steelblue',linewidth = '0.5', label = str2)
    ax.set_ylabel('PROBE [mV]', fontsize="large")
    ax.text(0.05, 0.95, 'c', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    # ax.set_ylim(-0.016, 0.016)

    ax = axes[3]
    ax.scatter(pos_pks_v1_v2*dt,diff_pks_v1_v2*dt*1e3, color ='darkorange',s = 3, label = 'Peaks')
    ax.scatter(pos_trh_v1_v2*td,diff_trh_v1_v2*dt*1e3, color = 'darkgreen',s = 3, label = 'Troughs')
    ax.set_ylabel('Delay [ms]', fontsize="large")
    # ax.axhline(y = 0, color = 'k', linestyle = '-')
    ax.set_xlim(start*dt, stop*dt) 
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    ax.text(0.05, 0.95, 'd', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    corr1, corr2, time_del = sliding_x_corr(sig1, sig2, sig1, sig2)
    ax = axes[4]
    ax.scatter(np.array(time_del)*dt, np.array(corr1)*dt*1e3,color = 'darkred', s = 3,label = 'X_corr')
    ax.set_ylabel('Delay [ms]', fontsize="large")
    # ax.axhline(y = 0, color = 'k', linestyle = '-')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    ax.text(0.05, 0.95, 'e', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    ax = axes[5]
    ax.scatter(pks_v1*dt, pktopk_pks_v1*1e3, color='k',s = 3, label=str1)
    ax.scatter(trh_v1*dt, pktopk_trh_v1*1e3, color='k',s = 3,)

    ax.scatter(pks_v2*dt, pktopk_pks_v2*1e3, color='steelblue',s = 3, label=str2)
    ax.scatter(trh_v2*dt, pktopk_trh_v2*1e3, color='steelblue',s = 3,)
    ax.set_xlabel(f'Time After {date[0]} [s]', fontsize=20)
    ax.set_ylabel('Pk-Pk [mV]', fontsize="large")
    # ax.set_ylim(0,0.032)
    ax.legend(loc='upper right')
    ax.text(0.05, 0.95, 'f', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    figure.tight_layout(w_pad=0.05, h_pad=0.05)
    figure.savefig(f"./Summer_Update/Spectrogram_Plots/2021-01-19-Trial{i}")
    plt.show()

def full_FFT_plot(sig1, sig2, sig3, sig4, start, stop):
    """
    Plot the sliding FFT of each of the 4 signals

    :param sig1: Wave 1
    :param sig2: Wave 2
    :param sig3: Wave 3
    :param sig4: Wave 4
    :param start: start time of the plots
    :param stop: stop time of the plots
    """
    start = int(start/dt)
    stop = int(stop/dt)
    sig1 = sig1[start:stop]
    sig2 = sig2[start:stop]
    sig3 = sig3[start:stop]
    sig4 = sig4[start:stop]
    
    # Number of samples in normalized_tone 
    #   Start time 

    window_size = 5000
    hop_size = 2500

    spectra1, spectra2, spectra3, spectra4, _ = sliding_fft(sig1,sig2,sig3,sig4, window_size, hop_size)
    
    xf = np.abs(fftfreq(window_size, 1 / samp_rt))
    fig, axes = plt.subplots(4, figsize=(12, 6), sharex=True)

    time_bins = np.linspace(start*dt, stop*dt, len(spectra1))
    ax = axes[0]
    im = ax.pcolormesh(time_bins, xf, spectra1.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    cb = fig.colorbar(im)
    cb.set_label("dV1", size="large")
    # ax.set_ylim(0,2000)
    ax.set_ylabel('Frequency [Hz]')

    time_bins = np.linspace(start*dt, stop*dt, len(spectra2))
    ax = axes[1]
    im = ax.pcolormesh(time_bins, xf, spectra2.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    cb = fig.colorbar(im)
    cb.set_label("dV2", size="large")
    # ax.set_ylim(0,2000)
    ax.set_ylabel('Frequency [Hz]')

    time_bins = np.linspace(start*dt, stop*dt, len(spectra3))
    ax = axes[2]
    im = ax.pcolormesh(time_bins, xf, spectra3.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    cb = fig.colorbar(im)
    cb.set_label("dV3", size="large")
    # ax.set_ylim(0,1000)
    ax.set_ylabel('Frequency [Hz]')

    time_bins = np.linspace(start*dt, stop*dt, len(spectra4))
    ax = axes[3]
    im = ax.pcolormesh(time_bins, xf, spectra4.T, shading='gouraud', cmap = 'jet', norm=mpl.colors.LogNorm())
    cb = fig.colorbar(im)
    cb.set_label("dV4", size="large")
    # ax.set_ylim(0,1000)
    ax.set_ylabel('Frequency [Hz]')
    fig.tight_layout(w_pad=0.05, h_pad=0.05)
    plt.show()

def sliding_x_corr(sig1, sig2, sig3, sig4):
    dev = window_size_x_corr
    time_delay1, x_corr1 = n_x_corr(sig2, sig1, int(len(sig1)/dev), time, 'dV2', 'dV1')
    time_delay2, x_corr2 = n_x_corr(sig4, sig3, int(len(sig1)/dev), time, 'dV4', 'dV3')
    corr1 = []
    corr2 = []
    time_del = []
    for i in range(len(time_delay1)):
        if x_corr1[i][1].max() >= 0.5 and x_corr2[i][1].max() >= 0.5:
            corr1.append(x_corr1[i][0]/dt)
            corr2.append(x_corr2[i][0]/dt)
            time_del.append(time_delay1[i]/dt)
    # Proform sliding X_corr on each signal
    # only keep X_corr that have accuracy above 0.95 or 0.9
    return corr1, corr2, time_del

def delay_plot(sig1,sigd1,sig2,sigd2):
    diff_pks, pos_pks,_,_,_,_,diff_trh, pos_trh,_,_,_,_ = frequency_filter(sig1, sig2, sig1, sig2, sig1)
    diffd_pks, posd_pks,_,_,_,_, diffd_trh, posd_trh,_,_,_,_ = frequency_filter(sigd1, sigd2, sigd1, sigd2, sigd1)
    delay = []
    fig= plt.figure()
    fig.suptitle("Figure 5", fontsize=25, weight='bold')
    gs = fig.add_gridspec(2, 2,  width_ratios=(4.3, 0.7), height_ratios=(1, 4),
                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                  wspace=0.02, hspace=0.01)
    # Create the Axes.
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(time[i],sig1*1e3,'k', label='dV1', linewidth = '0.5')
    ax1.plot(time[i],sig2*1e3,'steelblue', label = 'dV2', linewidth = '0.5')
    ax1.legend()
    ax1.set_ylabel('PROBE [mV]',fontsize=20)
    ax1.set_ylim(-6,6)
    ax1.set_xlabel(f'Time After {date[0]} [s]',fontsize=20)
    ax2 = ax1.twinx()
    ax2.scatter(np.array(pos_pks)*dt, np.array(diff_pks)*dt*1e3,color='darkred',s = 4)
    ax2.scatter(np.array(pos_trh)*dt, np.array(diff_trh)*dt*1e3,color='darkred',s = 4)
    ax2.scatter(np.array(posd_pks)*dt, np.array(diffd_pks)*dt*1e3,color='darkred', s = 4)
    ax2.scatter(np.array(posd_trh)*dt, np.array(diffd_trh)*dt*1e3,color='darkred', s = 4)
    # 0].plot(np.array(time_del)*dt, np.array(corr1)*dt,'.', label = 'x_corr')
    # 0].axhline(y = 0, color = 'purple', linestyle = '-')
    ax2.tick_params(axis='y', labelright = False)
    ax2.set_xlim(time[i].min(), time[i].max())
    ax2.tick_params(right = False)
    # ax2.set_ylim(-1, 1)
    ax2.grid(False)
    delay.extend(diff_pks)
    delay.extend(diff_trh)
    delay.extend(diffd_pks)
    delay.extend(diffd_trh)
    # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax2)
    ax_histy = fig.add_subplot(gs[:, 1], sharey=ax2)
    ax_histy.tick_params(axis="y", labelleft=False, labelright = True, labelcolor='darkred')
    ax_histy.yaxis.set_label_position("right")
    ax_histy.yaxis.tick_right()
    ax_histy.set_ylabel('Delay [ms]',fontsize=20, color='darkred')
    bins = np.linspace(-5, 5, 100)
    ax_histy.hist(np.array(delay)*dt*1e3,color='darkred',bins= bins, orientation='horizontal')

def angle_btw(vec1, vec2):
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    dot = np.dot(vec1, vec2)
    mag1 = np.sqrt(np.dot(vec1, vec1))
    mag2 = np.sqrt(np.dot(vec2, vec2))
    angle = np.arccos(dot/(mag1*mag2))
    angle = np.degrees(angle)
    return angle

# Start of main code
#################################################################################################################################
if __name__ == '__main__':
    #Reading in electric boom data from the cdf file
    plt.style.use('bmh')
    mpl.rcParams['font.family'] = 'Times New Roman'
    # mpl.rcParams['font.serif'] = ['Times New Roman']
    # rc('text', usetex=True)
    fld_vac_file = os.path.join(fld_vac, 'psp_fld_l2_dfb_dbm_{}_{}_v02.cdf'.format(file_type,date_wave))
    if os.path.exists(fld_vac_file) == False:
        print("Vac File Does not Exist or is not downloaded")
        exit()
    cdf_file = cdflib.CDF(fld_vac_file)
    info = cdf_file.cdf_info()
    zvars = info['zVariables']
    vac1 = cdf_file.varget(zvars[4])
    vac2 = cdf_file.varget(zvars[9])
    vac3 = cdf_file.varget(zvars[14])
    vac4 = cdf_file.varget(zvars[19])
    vac5 = cdf_file.varget(zvars[24])
    time_TT200 = cdf_file.varget(zvars[2])
    time_TT200 = np.array(time_TT200, dtype=np.int64)
    time = cdf_file.varget(zvars[3])

    #Sample rate
    samp_rt = len(time[0])/time[0][len(time[0])-1]
    nyquist = samp_rt / 2.0
    td = time[0,1]-time[0,0]

    # fig, ax = plt.subplots()
    # ax.plot(vac1)
    # plt.show()

    # Reading in dust file
    #------------------------------------------------------------------------------------
    # cdf_dust = cdflib.CDF(os.path.join(dpath, 'psp_fld_l3_dust_{}_v01.cdf'.format(data_date)))
    # info_dust = cdf_dust.cdf_info()
    # zvars_dust = info_dust['zVariables']
    # dust_epoch = cdf_dust.varget('psp_fld_l3_dust_V2_event_epoch')

    #Reading in magnetic field data from cdf file
    #------------------------------------------------------------------------------------
    fld_mag_file = os.path.join(fld_mag, 'psp_fld_l2_mag_SC_4_Sa_per_Cyc_{}_v02.cdf'.format(date_wave[0:8]))
    if os.path.exists(fld_mag_file) == False:
        print("Mag File Does not Exist or is not Downloaded")
        exit()
    cdf_mag = cdflib.CDF(fld_mag_file)
    info_mag = cdf_mag.cdf_info()
    zvars_mag = info_mag['zVariables']
    #Magnetic field components B_X = 0, B_Y = 1, B_Z = 2
    mag_fld = cdf_mag.varget(zvars_mag[1])

    #Rotate the magnetic field axis to match with V1-2 and V3-4 axis
    #------------------------------------------------------------------------------------
    rot_mat = [[np.cos(55*(math.pi/180)) , -np.sin(55*(math.pi/180)) , 0]  , [np.sin(55*(math.pi/180)) , np.cos(55*(math.pi/180)) , 0] , [0 , 0 , -1]]
    rot_mag_fld = np.dot(mag_fld, rot_mat)
    mag_fld_epoch = cdf_mag.varget(zvars_mag[0])

    #Solar wind csv file
    #Will do more with this later for now leave it be
    #------------------------------------------------------------------------------------
    # csvreader = csv.reader('/Users/jackr/Documents/Cattell Lab/Python/electrons_e1_e9 (version 1).csv')
    # sw_data = []
    # for row in csvreader:
    #    sw_data.append(row)
    # sw_data = sio.readsav(os.path.join(dpath, 'LFRV1-V2Ne_Tc_Thk_mimo_2021-01-03_2021-01-30.sav'))
    # sw_epoch = sw_data['epochl2']

    #Read in Solar Wind Data
    #------------------------------------------------------------------------------------
    swp_spi_file = os.path.join(swp_sw, 'psp_swp_spi_sf00_L3_mom_{}_v04.cdf'.format(date_wave[0:8]))
    if os.path.exists(swp_spi_file) == False:
        print("SW File Does not Exist or is not Downloaded")
        exit()
    sw_vel_file = cdflib.CDF(swp_spi_file)
    sw_info = sw_vel_file.cdf_info()
    sw_zvars = sw_info['zVariables']
    sw_vel_epoch = sw_vel_file.varget(sw_zvars[0])
    sw_vel = sw_vel_file.varget(sw_zvars[30])
    sw_vel = np.dot(sw_vel, rot_mat)

    #Loop Through all Trials
    #------------------------------------------------------------------------------------
    print('Starting Timing')
    for i in range(start_trial, len(vac1)):
        log_file = os.path.join(wpath, "Data from {} Trial {}.csv".format(date_wave, i))
        if os.path.exists(log_file):
            os.remove(log_file)
        date = cdflib.cdfepoch.to_datetime(time_TT200[i])
        v1 = vac1[i]
        v2 = vac2[i]
        v3 = vac3[i]
        v4 = vac4[i]
        v5 = vac5[i]
        dt = time[i][1]-time[i][0]


        #Mozer Method (Potential Subtraction)
        #------------------------------------------------------------------------------------   
        if pot_type == 'Moz':
            dv1 = v1 - (v3+v4)/2
            dv2 = v2 - (v3+v4)/2
            dv3 = v3 - (v1+v2)/2
            dv4 = v4 - (v1+v2)/2
            dv5 = v5 - (v1+v2+v3+v4)/4

        #Cattell Method (Potential Subtraction)
        #------------------------------------------------------------------------------------
        elif pot_type == 'Cat':
            vsc = (v1+v2+v3+v4)/4
            dv1 = v1 - vsc
            dv2 = v2 - vsc
            dv3 = v3 - vsc
            dv4 = v4 - vsc
            dv5 = v5 - vsc

        #No (Potential Subtraction)
        #------------------------------------------------------------------------------------
        elif pot_type == 'None':
            dv1 = v1
            dv2 = v2
            dv3 = v3
            dv4 = v4
            dv5 = v5

        else:
            print("Method Does not Exist")
            exit()

        dv2 = -dv2
        dv3 = -dv3
        dv5 = -dv5

        if plot_stuff == True:
            # full_FFT_plot(dv1, dv2, dv3, dv4, 0, len(dv1)*dt)
            frequency_plot(dv1, dv2, 0, len(dv1)*dt, 'dV1', 'dV2')
            frequency_plot(dv3, dv4, 0, len(dv1)*dt, 'dV3', 'dV4')
            # frequency_plot(dv1, dv5, 0, len(dv1)*dt, 'dV1', 'dV5')

        #Filter Waves (Using FIR filter to keep phase of wave)
        #------------------------------------------------------------------------------------
        if filt_type == 'Fir':
            dv1_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv1)
            dv2_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv2)
            dv3_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv3)
            dv4_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv4)
            dv5_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv5)
        else:
            dv1_filt = filter(low_cut_freq, high_cut_freq, filt_len, dv1)
            dv2_filt = filter(low_cut_freq, high_cut_freq, filt_len, dv2)
            dv3_filt = filter(low_cut_freq, high_cut_freq, filt_len, dv3)
            dv4_filt = filter(low_cut_freq, high_cut_freq, filt_len, dv4)
            dv5_filt = filter(low_cut_freq, high_cut_freq, filt_len, dv5)

        corr1, corr2, time_del = sliding_x_corr(dv2_filt, dv1_filt, dv4_filt, dv3_filt)
        
        fig, ax = plt.subplots()
        ax.plot(dv1_filt, label = 'dv1')
        ax.plot(dv2_filt, label = 'dv2')
        ax.plot(dv3_filt, label = 'dv3')
        ax.plot(dv4_filt, label = 'dv4')
        ax.plot(dv5_filt, label = 'dv5')
        ax.legend()
        plt.show()

        # Take Derivitive to find zero crossings
        deriv1, deriv2 = deriv(dv1_filt, dv2_filt,time)
        deriv3, deriv4 = deriv(dv3_filt, dv4_filt,time)
        deriv5, _ = deriv(dv5_filt, dv5_filt,time)

        # Filter Derivitive
        if filt_type == 'Fir':
            deriv1 = filter_fir(low_cut_freq, high_cut_freq, filt_len, deriv1)
            deriv2 = filter_fir(low_cut_freq, high_cut_freq, filt_len, deriv2)
            deriv3 = filter_fir(low_cut_freq, high_cut_freq, filt_len, deriv3)
            deriv4 = filter_fir(low_cut_freq, high_cut_freq, filt_len, deriv4)
            deriv5 = filter_fir(low_cut_freq, high_cut_freq, filt_len, deriv5)
        else:
            deriv1 = filter(low_cut_freq, high_cut_freq, filt_len, deriv1)
            deriv2 = filter(low_cut_freq, high_cut_freq, filt_len, deriv2)
            deriv3 = filter(low_cut_freq, high_cut_freq, filt_len, deriv3)
            deriv4 = filter(low_cut_freq, high_cut_freq, filt_len, deriv4)
            deriv5 = filter(low_cut_freq, high_cut_freq, filt_len, deriv5)

        # Use frequncy filter to find delays and positions
        #------------------------------------------------------------------------------------
        if v5_inclde == True:
            diff_pks_v1_v2, pos_pks_v1_v2, diff_pks_v3_v4, pos_pks_v3_v4,diff_pks_v1_v5, pos_pks_v1_v5,diff_trh_v1_v2, pos_trh_v1_v2, diff_trh_v3_v4, pos_trh_v3_v4,diff_trh_v1_v5, pos_trh_v1_v5 = frequency_filter(dv1_filt, dv2_filt, dv3_filt, dv4_filt, dv5_filt)
            diffd_pks_v1_v2, posd_pks_v1_v2, diffd_pks_v3_v4, posd_pks_v3_v4,diffd_pks_v1_v5, posd_pks_v1_v5, diffd_trh_v1_v2, posd_trh_v1_v2, diffd_trh_v3_v4, posd_trh_v3_v4, diffd_trh_v1_v5, posd_trh_v1_v5 = frequency_filter(deriv1, deriv2, deriv3, deriv4, deriv5)

        else:
            diff_pks_v1_v2, pos_pks_v1_v2, diff_pks_v3_v4, pos_pks_v3_v4,diff_pks_v1_v5, pos_pks_v1_v5,diff_trh_v1_v2, pos_trh_v1_v2, diff_trh_v3_v4, pos_trh_v3_v4,diff_trh_v1_v5, pos_trh_v1_v5 = frequency_filter(dv1_filt, dv2_filt, dv3_filt, dv4_filt, dv1_filt)
            diffd_pks_v1_v2, posd_pks_v1_v2, diffd_pks_v3_v4, posd_pks_v3_v4,diffd_pks_v1_v5, posd_pks_v1_v5, diffd_trh_v1_v2, posd_trh_v1_v2, diffd_trh_v3_v4, posd_trh_v3_v4, diffd_trh_v1_v5, posd_trh_v1_v5 = frequency_filter(deriv1, deriv2, deriv3, deriv4, deriv1)
    
        if plot_stuff == True:
            delay_plot(dv1_filt, deriv1, dv2_filt, deriv2)
            delay_plot(dv3_filt, deriv3, dv4_filt, deriv4)
            delay_plot(dv1_filt, deriv1, dv5_filt, deriv5)
            plt.show()

        # Put in array to be looped over
        # diff_v1_v2 = [diff_pks_v1_v2, diff_trh_v1_v2, diffd_pks_v1_v2, diffd_trh_v1_v2, corr1]
        # pos_v1_v2 = [pos_pks_v1_v2, pos_trh_v1_v2, posd_pks_v1_v2, posd_trh_v1_v2, time_del]
        # diff_v3_v4 = [diff_pks_v3_v4, diff_trh_v3_v4, diffd_pks_v3_v4, diffd_trh_v3_v4, corr2]
        # pos_v3_v4 = [pos_pks_v3_v4, pos_trh_v3_v4, posd_pks_v3_v4, posd_trh_v3_v4, time_del]
        # diff_v1_v5 = [diff_pks_v1_v5, diff_trh_v1_v5, diffd_pks_v1_v5, diffd_trh_v1_v5]
        # pos_v1_v5 = [pos_pks_v1_v5, pos_trh_v1_v5, posd_pks_v1_v5, posd_trh_v1_v5]

        diff_v1_v2 = [diff_pks_v1_v2, diff_trh_v1_v2, diffd_pks_v1_v2, diffd_trh_v1_v2]
        pos_v1_v2 = [pos_pks_v1_v2, pos_trh_v1_v2, posd_pks_v1_v2, posd_trh_v1_v2]
        diff_v3_v4 = [diff_pks_v3_v4, diff_trh_v3_v4, diffd_pks_v3_v4, diffd_trh_v3_v4]
        pos_v3_v4 = [pos_pks_v3_v4, pos_trh_v3_v4, posd_pks_v3_v4, posd_trh_v3_v4]
        diff_v1_v5 = [diff_pks_v1_v5, diff_trh_v1_v5, diffd_pks_v1_v5, diffd_trh_v1_v5]
        pos_v1_v5 = [pos_pks_v1_v5, pos_trh_v1_v5, posd_pks_v1_v5, posd_trh_v1_v5]

        type_delay = ['Peaks', 'Troughs', 'Zero Rising', 'Zero-Falling', 'Cross-Correlation']

        # Open .csv file where data will be output to
        with open(log_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(["Date: {} to {}".format(date[0], date[len(date)-1]),
                             'Time of V1-V2 Delay [s after {}]'.format(date[0]), 'V1-V2 Delay [s]', 
                             'Time of V3-V4 Delay [s after {}]'.format(date[0]), 'V3-V4 Delay [s]', 
                             'Time of Vsc-V5 Delay [s after {}]'.format(date[0]), 'Vsc-V5 Delay [s]', 
                             'Angle of Wave wrt V1 Boom [Deg]', 'Angle of Wave wrt V5 Boom [Deg]',
                             'Velocity of Wave along V1-2 Boom [km/s]', 'Velocity of Wave along V3-4 Boom [km/s]', 'Velocity of Wave along V5 Boom [km/s]', 
                             'Angle btw wave and the mag fld in X-Y Plane[Degrees]', 'Angle btw wave and the mag fld out of X-Y Plane[Degrees]',
                             'Angle between solar wind and wave [Degrees]', 
                             'Angle between the magnetic field and V1 Boom [Degrees]', 'Angle between the magnetic field and V5 [Degrees]', 
                             'Magnitude of the magnetic field [nT]', 'Magnitude of the magnetic field in X-Y [nT]', 
                             'Solar Wind Speed in X-Y Plane [km/s]', 'Sample Rate [Samples/s]', 'V5 Included?'])
            writer.writerow(['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '','','','','', samp_rt, v5_inclde])
            f = 0
            theta_arr = np.zeros(len(diff_v1_v2[0])+len(diff_v1_v2[1])+len(diff_v1_v2[2])+len(diff_v1_v2[3]))
            vel_arr = np.zeros(len(diff_v1_v2[0])+len(diff_v1_v2[1])+len(diff_v1_v2[2])+len(diff_v1_v2[3]))
            angle_out_arr = np.zeros(len(diff_v1_v2[0])+len(diff_v1_v2[1])+len(diff_v1_v2[2])+len(diff_v1_v2[3]))

            # loop over all delays computing angles and velocities and adding them to the .csv
            for k in range(len(diff_v1_v2)):
                writer.writerow([type_delay[k], '', '', '', '', '', '', '', '', '', '','','','','', '', '','','','',''])
                for j in range(len(diff_v1_v2[k])):
                    t_12 = diff_v1_v2[k][j]*dt
                    t_34 = diff_v3_v4[k][j]*dt
                    t_5 = diff_v1_v5[k][j]*dt

                    mag_fld_index = find_nearest(mag_fld_epoch, time_TT200[i][int(pos_v1_v2[k][j])])
                    mag_fld_comp = rot_mag_fld[mag_fld_index]
                    mag_fld_angle = angle_wrt_v1(mag_fld_comp[0], mag_fld_comp[1])

                    # Angle of the magnetic field out of the X-Y plane
                    #------------------------------------------------------------------------------------
                    mag_vector = np.sqrt((mag_fld_comp[0]**2)+(mag_fld_comp[1]**2)+(mag_fld_comp[2]**2))
                    mag_vector_in_x_y = np.sqrt((mag_fld_comp[0]**2)+(mag_fld_comp[1]**2))
                    # Angle wrt V5 Boom
                    mag_angle_out = np.arccos(mag_fld_comp[2]/mag_vector)*(180/np.pi)

                    #Solar wind angle
                    #------------------------------------------------------------------------------------
                    sw_vel_index = find_nearest(sw_vel_epoch, time_TT200[i][int(pos_v1_v2[k][j])])
                    sw_vel_comp = sw_vel[sw_vel_index]
                    sw_vel_angle = angle_wrt_v1(sw_vel_comp[0], sw_vel_comp[1])
                    sw_speed = np.sqrt(sw_vel_comp[1]**2+sw_vel_comp[0]**2)
                    # print(sw_vel)

                    # angle_out[f] = t_12
                    # angle_out[f] = t_5
                    # Angle between wave and V1 boom
                    #------------------------------------------------------------------------------------
                    # print(f"Pre-Angle: {np.arctan(t_34/t_12)*(180/np.pi)}")
                    angle_wave_v1 = angle_wrt_v1(t_12, t_34, wave_angle=False)

                    # Angle wrt V5 Boom
                    #------------------------------------------------------------------------------------
                    angle_wave_v5 = np.arccos(t_5/np.sqrt(t_12**2+(t_34*np.cos(angle_wave_v1*(np.pi/180)))**2+t_5**2))*(180/np.pi) # Look at how these angles were determined

                    # Calculating Wave Speed
                    #------------------------------------------------------------------------------------
                    if t_12 == 0:
                        vel_12 = 0*(u.m/u.s)
                    else:
                        vel_12 = abs(boom_len*np.cos(angle_wave_v1*(np.pi/180)))/t_12*(u.m/u.s)
                    if t_34 == 0:
                        vel_34 = 0*(u.m/u.s)
                    else:
                        vel_34 = abs(boom_len*np.sin(angle_wave_v1*(np.pi/180)))/t_34*(u.m/u.s)
                    if t_5 == 0:
                        vel_5 = 0*(u.m/u.s)
                    else:
                        vel_5 = abs(6*np.cos(angle_wave_v5*(np.pi/180)))/t_5*(u.m/u.s)
                    # print(vel_5)

                    vel_12 = vel_12.to(u.km/u.s)
                    vel_34 = vel_34.to(u.km/u.s)
                    vel_5 = vel_5.to(u.km/u.s)
                    # vel_5 = np.sqrt(vel_12**2+vel_34**2)
                    v1_vec = np.array([1,0,0])
                    v5_vec = np.array([0,0,1])
                    delay_vec = np.array([t_12,t_34,t_5])
                    vel_vec = np.array([vel_12.value,vel_34.value,vel_5.value])
                    sw_vec = np.array([sw_vel_comp[0], sw_vel_comp[1], sw_vel_comp[2]])
                    # Wave - Sw or Sw - Wave?
                    vel_plas_vec = sw_vec - vel_vec
                    B_vec = np.array([mag_fld_comp[0], mag_fld_comp[1], mag_fld_comp[2]])
                    
                    angle_wave_mag_in = abs(angle_wave_v1-mag_fld_angle)

                    print(f"Delay {delay_vec}")
                    print(f"Vel Plasma {vel_plas_vec}")
                    print(f"Vel {vel_vec}")
                    print(f"btw delay and B0 {angle_btw(delay_vec, B_vec)}")
                    print(f"btw delay and B0 {angle_wave_mag_in}")
                    print(f"btw vel and B0 {angle_btw(vel_vec, B_vec)}")
                    print(f"btw sw and B0 {angle_btw(sw_vec, B_vec)}")
                    print(f"btw plas and B0 {angle_btw(vel_plas_vec, B_vec)}")

                    # Angle between the magnetic field and Wave
                    #------------------------------------------------------------------------------------
                    angle_wave_mag_in = abs(angle_wave_v1-mag_fld_angle)
                    angle_out_arr[f] = angle_wave_mag_in
                    angle_wave_mag_out = abs(angle_wave_v5-mag_angle_out)

                    # save data to array for 2d hist
                    theta_arr[f] = angle_wave_v1
                    vel_arr[f] = np.sqrt(vel_12.value**2+(vel_34.value*np.cos(angle_wave_v1*(np.pi/180)))**2)

                    # Angle between solar wind and Wave
                    #------------------------------------------------------------------------------------
                    sw_angle = abs(angle_wave_v1-sw_vel_angle)

                    # Write Data to csv
                    #------------------------------------------------------------------------------------
                    if v5_inclde == True:
                        writer.writerow(['', 
                                         pos_v1_v2[k][j]*dt, t_12, 
                                         pos_v3_v4[k][j]*dt, t_34, 
                                         pos_v1_v5[k][j]*dt, t_5, 
                                         angle_wave_v1, angle_wave_v5,
                                         vel_12.value, vel_34.value, vel_5.value,
                                         angle_wave_mag_in, angle_wave_mag_out,
                                         sw_angle, mag_fld_angle, mag_angle_out, mag_vector, mag_vector_in_x_y, sw_speed, ''])
                    else:
                        writer.writerow(['', 
                                         pos_v1_v2[k][j]*dt, t_12, 
                                         pos_v3_v4[k][j]*dt, t_34, 
                                         '', '', 
                                         angle_wave_v1, '',
                                         vel_12.value, vel_34.value, '',
                                         angle_wave_mag_in, '',
                                         sw_angle, mag_fld_angle, mag_angle_out, mag_vector, mag_vector_in_x_y, sw_speed, ''])
                    f=f+1

            if len(theta_arr)>10 and plot_stuff==True:
                polar_2d_hist(theta_arr*(np.pi/180),vel_arr, mag_fld_angle*(np.pi/180), mag_vector_in_x_y, sw_vel_angle*(np.pi/180), sw_speed)

            theta_x_corr = np.zeros(len(corr1))
            for j in range(len(corr1)):
                t_12 = corr1[j]*dt
                t_34 = corr2[j]*dt
                angle_wave_v1 = angle_wrt_v1(t_12, t_34, wave_angle=True)
                mag_fld_index = find_nearest(mag_fld_epoch, time_TT200[i][int(time_del[j])])
                mag_fld_comp = rot_mag_fld[mag_fld_index]
                mag_fld_angle = angle_wrt_v1(mag_fld_comp[0], mag_fld_comp[1])
                theta_x_corr[j] = abs(angle_wave_v1-mag_fld_angle)

            fig, ax1 = plt.subplots()
            colors = ['steelblue','darkred']
            fig.suptitle("Figure 6", fontsize=25, weight='bold')
            ax2 = ax1.twinx()
            bins = np.linspace(0, 360, 75)
            ax1.hist([angle_out_arr, theta_x_corr], bins = bins, color=colors)
            n,bins,patches = ax1.hist([angle_out_arr, theta_x_corr],bins = bins)
            ax1.cla()
            # ax.hist(angle_out_arr, bins = bins, label='Mag Angle Out')
            # ax.hist(theta_x_corr, bins=bins)
            width = (bins[1]-bins[0])*0.5
            bins_shifted = bins+width
            ax1.bar(bins[:-1], n[0],width,align='edge',color = colors[0])
            ax2.bar(bins_shifted[:-1], n[1],width,align='edge',color = colors[1])


            # ax.set_ylabel('Counts', fontsize=20)
            # ax.set_xlabel('Angle Between Wave and B0 [Deg]', fontsize=20)
            ax1.set_ylabel("Peak Method", color=colors[0],fontsize=20)
            ax2.set_ylabel("X-Correlation Method", color=colors[1],fontsize=20)
            ax1.tick_params('y',colors=colors[0])
            ax2.tick_params('y',colors=colors[1])
            ax1.grid(False)
            ax2.grid(False)
            ax1.set_xlim(0, 360)
            ax1.set_xlabel("Angle of Wave with Respect to B0",fontsize=20)
            plt.tight_layout()
            plt.show()
    
        print('Trial {} out of {}'.format(i, len(vac1)-1))
    print('Completed')
