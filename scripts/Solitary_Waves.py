#IMPORTS
import cdflib
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import math
from astropy import constants as const
import statistics
import os
from scipy.signal import firwin, freqz
from astropy import units as u
# from pytplot import get_data

#Change these to switch the datafile being run
#------------------------------------------------------------------------------------
#date for psp_fld_I2_mag (magnetic field) and psp_swp_spi (solar wind) files
#data_date = '20210119'

#date for psp_fld_I2_dfb_dbm_vac (wave) file
#date_wave = data_date + '00'

date_wave = ''
while len(date_wave) != 10:
  date_wave = input('What File would you like to Run: ')
  if len(date_wave) != 10:
    print("Enter Valid Time")

#Bandpass Filter Parameters
#Low Cut Frequency
low_cut_freq = 100
#High Cut Frequency
high_cut_freq = 900
#Filter length
filt_len = 601

# Time V5?
v5_inclde = False
# vdc or vac?
file_type = 'vac'

#Which Trial would you like to start with?
start_trial = 0
#------------------------------------------------------------------------------------

dpath = os.path.dirname(os.path.realpath(__file__))
wpath = os.path.join(dpath, 'data_output')
if os.path.exists(wpath) == False:
    os.mkdir(wpath)
wpath = os.path.join(wpath, 'Date {}'.format(date_wave))
figpath = os.path.join(wpath, 'Full Figure')
figpathx = os.path.join(wpath, 'X_corr Figures')
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

#FUNCTIONS
#------------------------------------------------------------------------------------
def find_nearest(array, value):
    """
    Find the nearest value in the array to the value given

    :param array: array of numbers
    :param value: 
    :return: The index of the array with the number closest to value given
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def angle_btw_mag_ele(ele_angle, mag_angle, mag_x, mag_y, ele_x, ele_y, i):
    """
    Finding the angle between the magnetic field vector and electic wave

    :param ele_angle: Angle between the electric wave and V1-2 boom
    :param mag_angle: Angle between magnetci field and V1-2 boom
    :param mag_x: x component of the magnetic field
    :param mag_y: y component of the magntiic field
    :param ele_x: Time dalay of the V1-2 boom
    :param ele_y: Time delay of the V3-4 boom
    :return: Angle between the magneitc field and electric wave
    """
    if (mag_x>=0) and (ele_x>=0):
        if(ele_angle<mag_angle):
            angle = mag_angle-ele_angle
        else:
            angle = ele_angle-mag_angle
    elif ((mag_x>=0) and (ele_x<=0)):
        angle = 180 - abs(abs(mag_angle) - abs(ele_angle))
    elif((mag_x<=0) and (ele_x>=0)):
        angle = 180 - abs(abs(mag_angle) - abs(ele_angle))
    elif (mag_x<=0) and (ele_x<=0):
        if(ele_angle<mag_angle):
            angle = mag_angle-ele_angle
        else:
            angle = ele_angle-mag_angle
    else:
        angle = np.nan
    return angle

def x_corr(array1, array2, start, end, array1_str, array2_str, time, date):
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
    array1_snip = array1[start:end]
    array2_snip = array2[start:end]
    time = time[0][start:end]
    f1 = interpolate.interp1d(time, array1_snip, fill_value="extrapolate", kind = 'quadratic')
    f2 = interpolate.interp1d(time, array2_snip, fill_value="extrapolate", kind = 'quadratic')

    time_inter = np.linspace(start*dt, end*dt, len(time)*10)
    array1 = f1(time_inter)
    array2 = f2(time_inter)

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(time, array1_snip, 'o', label = array1_str)
    ax[0].plot(time_inter, array1, '-x')
    ax[1].plot(time, array2_snip, 'o',label = array2_str)
    ax[1].plot(time_inter, array2, '-x')
    ax[1].set_xlabel('t [s]')
    ax[0].set_ylabel('V [mV]')
    ax[1].set_ylabel('V [mV]')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    norm1 = (array1 - np.mean(array1)) / (np.std(array1) * len(array1))
    norm2 = (array2 - np.mean(array2)) / (np.std(array2))
    correlation = signal.correlate(norm1, norm2, mode="full", method = "fft")
    lags = signal.correlation_lags(len(norm1),len(norm2), mode = "full")
    lags = lags*dt
    xmax = lags[np.argmax(correlation)]
    pks, _ = signal.find_peaks(-correlation)

    fig, ax = plt.subplots()
    fig.suptitle(f"{array1_str} and {array2_str}")
    ax.plot(lags, correlation, '.')
    ax.plot(lags[pks], correlation[pks],'x')
    ax.set_xlabel('Delay [s]')
    ax.set_ylabel('Correlation')
    plt.show()

    fig, ax = plt.subplots(2)
    fig.suptitle(f"{array1_str} and {array2_str}")
    ax[0].plot(lags, correlation, '.')
    ax[0].set_xlabel('Delay [s]')
    ax[0].set_ylabel('Correlation')

    if len(pks)>1:
        print(pks)
        upr = find_nearest(pks[pks>np.argmax(correlation)], np.argmax(correlation))
        lwr = find_nearest(pks[pks<np.argmax(correlation)], np.argmax(correlation))
        upr = pks[pks>np.argmax(correlation)][upr]
        lwr = pks[pks<np.argmax(correlation)][lwr]
        correlation = correlation[lwr:upr]
        lags = lags[lwr:upr]

    errr_det = 1.2*(1-correlation[np.argmax(correlation)])
    vals = lags[(1-correlation)<errr_det]
    lwr_bnd = vals.min()
    upr_bnd = vals.max()
    upr_err = abs(lags[np.argmax(correlation)]-upr_bnd)
    lwr_err = abs(lags[np.argmax(correlation)]-lwr_bnd)
    err = (lwr_err+upr_err)/2
    ax[1].plot(lags, correlation, '.')
    ax[1].axvline(x = lwr_bnd, color = 'b', label = 'Low Error')
    ax[1].axvline(x = upr_bnd, color = 'b', label = 'High Error')
    ax[1].set_xlabel('Delay')
    ax[1].set_ylabel('Correlation')
    ax[1].legend()
    plt.show()
    return xmax, correlation, err

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
            angle = 95
        else:
            angle = -85
    else:
        frac = t_34/t_12
        minum = abs((np.sin((in_angle+5-15)*(math.pi/180))/np.cos((in_angle-15)*math.pi/180))-frac)
        for i in range(1, 30):
            diff = abs((np.sin((in_angle+5-15+i)*(math.pi/180))/np.cos((in_angle-15+i)*math.pi/180))-frac)
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

def sw_speed(t_12, t_34, angle):
    """
    Calculating the speed of the wave

    :param t_12: Time lag between the V1 boom and V2 boom
    :param t_34: Time lag between the V3 boom and V4 boom
    :param angle: Angle of the incoming wave with respect to the V1 V2 boom
    :return: Speed of the incoming wave
    """
    #angle in degrees
    #is 2m the whole boom or just half?
    if t_12 < 0 and t_34 < 0 or t_12 > 0 and t_34 > 0:
        vel_12 = (boom_len*np.cos(angle))/t_12
        vel_34 = (boom_len*np.cos(85-angle))/t_34
    else:
        vel_12 = (boom_len*np.cos(angle))/t_12
        vel_34 = (boom_len*np.cos(95-angle))/t_34
    vel_tot = np.sqrt((vel_12**2)+(vel_34**2))
    #vel_tot is an approxemation since the V_12 and V_34 booms are not orthogonal
    return vel_tot

def find_phase(corr, fs, peak_width = 11):

    """
    Finds the phase, frequency, and time delay using the cross corrilation

    :param corr: Cross corrrilation of the two waves
    :param fs: Sample rate
    :param peak_width: Width of the peak which is set to 11
    :return: The phase(deg), frequency(Hz), and delay(sec) of the two waves
    """
    corr_max = signal.find_peaks(corr, width=peak_width)[0]
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
    #Not Used in This version of the code
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
    for i in range(0,num_x_corr):
    #    fig, ax = plt.subplots()
        x_corr_lst[i] = x_corr(array1, array2, start, end, arr_str_1, arr_str_2, time, date)
        x_corr_2[i] = find_phase(x_corr_lst[i][1], samp_rt)
    #    time_lst = time[1] + x_corr_lst[i][0]
    #    time_2 = time[1] + x_corr_2[i][2]
    #    ax.plot(time[1][start:end],array1[start:end], label = arr_str_1)
    #    ax.plot(time[1][start:end],array2[start:end], label = arr_str_2)
    #    ax.plot(time_lst[start:end], array2[start:end], label = arr_str_2 + " original")
    #    ax.plot(time_2[start:end], array2[start:end], label = arr_str_2 + " mike")
    #    plt.legend()
    #    plt.show()
        start = start + par_array
        end = end + par_array
    return x_corr_lst, x_corr_2

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
    band = FIRBandPass(width, [low_cut, high_cut], fs=nyquist*2)
    filt = band.filter(wave)
    return filt

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

#Constants
#------------------------------------------------------------------------------------
#Boom length is about ~3.5m
boom_len = 3.5
#------------------------------------------------------------------------------------
if __name__ == '__main__':
    log_file = os.path.join(wpath, "Data from {}.txt".format(date_wave))
    print('Would you like to clear the log file?')
    del_log = ''
    while del_log != 'y' and del_log != 'n':
        del_log = input('[y]/n: ')
    if del_log == 'y':
        if os.path.exists(log_file):
            os.remove(log_file)

    #Reading in electric boom data from the cdf file
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
    dt = time[0,1]-time[0,0]

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
    rot_mat = [[np.cos(55*(math.pi/180)) , -np.sin(55*(math.pi/180)) , 0]  , 
               [np.sin(55*(math.pi/180)) , np.cos(55*(math.pi/180)) , 0] , [0 , 0 , -1]]
    rot_mag_fld = np.dot(mag_fld, rot_mat)
    mag_fld_epoch = cdf_mag.varget(zvars_mag[0])

    #Solar wind csv file
    #Will do more with this later for now leave it be
    #------------------------------------------------------------------------------------
    #csvreader = csv.reader('/Users/jackr/Documents/Cattell Lab/Python/electrons_e1_e9 (version 1).csv')
    #sw_data = []
    #for row in csvreader:
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

    print("Trials: ")
    for i in range(len(vac1)):
        date = cdflib.cdfepoch.to_datetime(time_TT200[i])
        print(f"Trial {i}: {date[0]} to {date[len(date)-1]}")
    print('Is there a Spicific Trial you want to see?')
    spec_trial = ''
    while spec_trial != 'y' and spec_trial != 'n':
        spec_trial = input('[y]/n: ')
    if spec_trial == 'y':
        start_trial = -1
        while 0>start_trial or start_trial>len(vac1)-1:
            start_trial = int(input(f"Which one? 0 - {len(vac1)-1}: "))

    #Loop Through all Trials
    #------------------------------------------------------------------------------------
    print('\nThere are {} Trials on this Day (indexed from 0 to {}). Start Trial: {}'.format(len(vac1),len(vac1)-1, start_trial))
    for i in range(start_trial, len(vac1)):
        date = cdflib.cdfepoch.to_datetime(time_TT200[i])
        v1 = vac1[i]
        v2 = vac2[i]
        v3 = vac3[i]
        v4 = vac4[i]
        v5 = vac5[i]

        #Mozer Method
        #------------------------------------------------------------------------------------   
        dv1 = v1 - (v3+v4)/2
        dv2 = v2 - (v3+v4)/2
        dv3 = v3 - (v1+v2)/2
        dv4 = v4 - (v1+v2)/2
        dv5 = v5 - (v1+v2+v3+v4)/2

        #Cattell Method
        #------------------------------------------------------------------------------------
        #vsc = (v1+v2+v3+v4)/4
        #dv1 = v1 - vsc
        #dv2 = v2 - vsc
        #dv3 = v3 - vsc
        #dv4 = v4 - vsc

        # dv1 = v1
        # dv2 = v2
        # dv3 = v3
        # dv4 = v4
        # dv5 = v5


        #Filter Waves
        #------------------------------------------------------------------------------------
        # dv1_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv1)
        # dv2_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv2)
        # dv3_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv3)
        # dv4_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv4)
        # dv5_filt = filter_fir(low_cut_freq, high_cut_freq, filt_len, dv5)

        dv1_filt = dv1
        dv2_filt = dv2
        dv3_filt = dv3
        dv4_filt = dv4
        dv5_filt = dv5

        # Should be -v2 and -v3 but doesnt line up
        dv2_filt = dv2_filt
        dv3_filt = dv3_filt
        dv5_filt = -dv5_filt
        vsc = (dv1_filt+dv2_filt+dv3_filt+dv4_filt)/4

        #Calculating Magnetic Field Angle
        #------------------------------------------------------------------------------------
        mag_fld_index = find_nearest(mag_fld_epoch, time_TT200[i][0])
        mag_fld_comp = rot_mag_fld[mag_fld_index]
        mag_fld_angle = angle_wrt_v1(mag_fld_comp[0], mag_fld_comp[1])

        mag_vector = np.sqrt((mag_fld_comp[0]**2)+(mag_fld_comp[1]**2)+(mag_fld_comp[2]**2))
        mag_vector_in_x_y = np.sqrt((mag_fld_comp[0]**2)+(mag_fld_comp[1]**2))
        # Angle wrt V5 Boom
        mag_angle_out = np.arccos(mag_fld_comp[2]/mag_vector)*(180/np.pi)


        #Solar wind angle
        #------------------------------------------------------------------------------------
        sw_vel_index = find_nearest(sw_vel_epoch, time_TT200[i][0])
        sw_vel_comp = sw_vel[sw_vel_index]
        sw_vel_angle = angle_wrt_v1(sw_vel_comp[0], sw_vel_comp[1])
        sw_speed = np.sqrt(sw_vel_comp[1]**2+sw_vel_comp[0]**2+sw_vel_comp[2]**2) * (u.km/u.s)

        with open(log_file, 'a') as f: 
            f.write('################################## Trial {} ##################################\n'.format(i))
            f.write("Date: {} to {}".format(date[0], date[len(date)-1]))
            f.write('\n'*2)

            f.write("Sample Rate [Samples/s] \n")
            f.write('{}'.format(samp_rt))
            f.write('\n'*3)

        sav_fig_v12 = os.path.join(figpath, "V1 and V2 Trial {} date {}.png".format(i, date_wave))
        sav_fig_v34 = os.path.join(figpath, "V3 and V4 Trial {} date {}.png".format(i, date_wave))


        if os.path.exists(sav_fig_v12):
            os.remove(sav_fig_v12)

        if os.path.exists(sav_fig_v34):
            os.remove(sav_fig_v34)

        arr_v = range(0, len(date))
        
        plt.plot(arr_v*dt,dv1_filt,label = "V1")
        plt.plot(arr_v*dt,dv2_filt,label = "-V2")
        plt.title('V1 and -V2 Trial {}'.format(i))
        plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
        plt.legend()
        plt.savefig(sav_fig_v12)
        plt.close()

        plt.plot(arr_v*dt,dv3_filt,label = "-V3")
        plt.plot(arr_v*dt,dv4_filt,label = "V4")
        plt.title('-V3 and V4 Trial {}'.format(i))
        plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
        plt.legend()
        plt.savefig(sav_fig_v34)
        plt.close()

        another_sec = 'y'
        j = 0
        while another_sec == 'y':
            #Plotting V1-4 vs time and Saving to 'data output' Directory (Add way to time multiple sections)
            #------------------------------------------------------------------------------------
            sav_fig_v12_xcorr = os.path.join(figpathx, "V1 and V2 Trial {} date {} X_corr sec {}.png".format(i, date_wave, j))
            sav_fig_v34_xcorr = os.path.join(figpathx, "V3 and V4 Trial {} date {} X_corr sec {}.png".format(i, date_wave,j))
            if v5_inclde == True:
                sav_fig_v5 = os.path.join(figpath, "Vsc and V5 Trial {} date {}.png".format(i, date_wave))
                sav_fig_v5_xcorr = os.path.join(figpathx, "Vsc and V5 Trial {} date {} X_corr sec {}.png".format(i, date_wave, j))
                if os.path.exists(sav_fig_v5):
                    os.remove(sav_fig_v5)

            #V1-2
            input('Which Part of the V1-2 Wave Would you Like to Correlate? Press Enter When Ready to Continue to See Wavefrom.')
            plt.plot(arr_v*dt,dv1_filt,label = "V1")
            plt.plot(arr_v*dt,dv2_filt,label = "-V2")
            plt.title('V1 and -V2 Trial {}'.format(i))
            plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
            plt.legend()
            plt.show(block=False)
            out = ''

            # Add way to do multiple time sections
            #########################################################
            while out != 'y':
                if os.path.exists(sav_fig_v12_xcorr):
                    os.remove(sav_fig_v12_xcorr)
                print('Which Part of the V1-2 Wave Would you Like to Correlate?')
                # Change this to date instead of array shifts
                start_point_v12 = -1
                end_point_v12 = -1
                while start_point_v12>=end_point_v12:
                    start_point_v12 = -1
                    end_point_v12 = -1
                    while start_point_v12<0 or start_point_v12>len(dv1_filt):
                        start_point_v12 = float(input('Starting Point [s]: '))
                        start_point_v12 = int(start_point_v12/dt)
                        if start_point_v12<0 or start_point_v12>len(dv1_filt):
                            print("Selected Time Out of Range")

                    while end_point_v12<0 or end_point_v12>len(dv1_filt):
                        end_point_v12 = float(input('End Point [s]: '))
                        end_point_v12 = int(end_point_v12/dt)
                        if end_point_v12<0 or end_point_v12>len(dv1_filt):
                            print("Selected Time Out of Range")

                    if start_point_v12>=end_point_v12:
                        print("Invalid Time Range Selected")

                plt.plot(arr_v[start_point_v12:end_point_v12]*dt,dv1_filt[start_point_v12:end_point_v12],'.',label = "V1_Sec")
                plt.plot(arr_v[start_point_v12:end_point_v12]*dt,dv2_filt[start_point_v12:end_point_v12],'.',label = "-V2_Sec")
                plt.xlim(start_point_v12*dt,end_point_v12*dt)
                plt.title('V1 and -V2 Trial {} X_Corr'.format(i))
                plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
                plt.legend()
                plt.savefig(sav_fig_v12_xcorr)
                plt.show(block=False)

                print('Is this the section you want to Correlate?')
                out = ''
                while out != 'y' and out != 'n':
                    out = input('[y]/n: ')
            plt.close()

            #V3-4
            input('Which Part of the V3-4 Wave Would you Like to Correlate? Press Enter When Ready to Continue to See Wavefrom.')
            plt.plot(arr_v*dt,dv3_filt,label = "-V3")
            plt.plot(arr_v*dt,dv4_filt,label = "V4")
            plt.title('-V3 and V4 Trial {}'.format(i))
            plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
            plt.legend()
            plt.show(block=False)
            out = ''

            while out != 'y':
                if os.path.exists(sav_fig_v34_xcorr):
                    os.remove(sav_fig_v34_xcorr)
                print('Which Part of the V3-4 Wave Would you Like to Correlate?')
                # Change this to date instead of array shifts
                start_point_v34 = -1
                end_point_v34 = -1
                while start_point_v34>=end_point_v34:
                    start_point_v34 = -1
                    end_point_v34 = -1
                    while start_point_v34<0 or start_point_v34>len(dv3_filt):
                        start_point_v34 = float(input('Starting Point [s]: '))
                        start_point_v34 = int(start_point_v34/dt)
                        if start_point_v34<0 or start_point_v34>len(dv3_filt):
                            print("Selected Time Out of Range")

                    while end_point_v34<0 or end_point_v34>len(dv3_filt):
                        end_point_v34 = float(input('End Point [s]: '))
                        end_point_v34 = int(end_point_v34/dt)
                        if end_point_v34<0 or end_point_v34>len(dv3_filt):
                            print("Selected Time Out of Range")

                    if start_point_v34>=end_point_v34:
                        print("Invalid Time Range Selected")

                plt.plot(arr_v[start_point_v34:end_point_v34]*dt,dv3_filt[start_point_v34:end_point_v34],'.',label = "-V3_Sec")
                plt.plot(arr_v[start_point_v34:end_point_v34]*dt,dv4_filt[start_point_v34:end_point_v34],'.',label = "V4_sec")
                plt.xlim(start_point_v34*dt,end_point_v34*dt)
                plt.title('-V3 and V4 Trial {} X_Corr'.format(i))
                plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
                plt.legend()
                plt.savefig(sav_fig_v34_xcorr)
                plt.show(block=False)

                print('Is this the section you want to Correlate?')
                out = ''
                while out != 'y' and out != 'n':
                    out = input('[y]/n: ')
            plt.close()

            if v5_inclde == True:
                #Vsc-5
                input('Which Part of the Vsc-5 Wave Would you Like to Correlate? Press Enter When Ready to Continue to See Wavefrom.')
                plt.plot(arr_v*dt,vsc,label = "Vsc")
                plt.plot(arr_v*dt,dv5_filt,label = "V5")
                plt.title('-Vsc and V5 Trial {}'.format(i))
                plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
                plt.legend()
                plt.savefig(sav_fig_v5)
                plt.show(block=False)
                out = ''

                while out != 'y':
                    if os.path.exists(sav_fig_v5_xcorr):
                        os.remove(sav_fig_v5_xcorr)
                    print('Which Part of the Vsc-5 Wave Would you Like to Correlate?')
                    # Change this to date instead of array shifts
                    start_point_v5 = -1
                    end_point_v5 = -1
                    while start_point_v5>=end_point_v5:
                        start_point_v5 = -1
                        end_point_v5 = -1
                        while start_point_v5<0 or start_point_v5>len(dv5_filt):
                            start_point_v5 = float(input('Starting Point [s]: '))
                            start_point_v5 = int(start_point_v5/dt)
                            if start_point_v5<0 or start_point_v5>len(dv5_filt):
                                print("Selected Time Out of Range")

                        while end_point_v5<0 or end_point_v5>len(dv5_filt):
                            end_point_v5 = float(input('End Point [s]: '))
                            end_point_v5 = int(end_point_v5/dt)
                            if end_point_v5<0 or end_point_v5>len(dv5_filt):
                                print("Selected Time Out of Range")

                        if start_point_v5>=end_point_v5:
                            print("Invalid Time Range Selected")

                    plt.plot(arr_v[start_point_v5:end_point_v5]*dt,vsc[start_point_v5:end_point_v5],label = "Vsc_Sec")
                    plt.plot(arr_v[start_point_v5:end_point_v5]*dt,dv5_filt[start_point_v5:end_point_v5],label = "-V5_sec")
                    plt.xlim(start_point_v5*dt,end_point_v5*dt)
                    plt.title('Vsc and -V5 Trial {} X_Corr'.format(i))
                    plt.xlabel("{} to {}".format(date[0], date[len(date)-1]))
                    plt.legend()
                    plt.savefig(sav_fig_v5_xcorr)
                    plt.show(block=False)

                    print('Is this the section you want to Correlate?')
                    out = ''
                    while out != 'y' and out != 'n':
                        out = input('[y]/n: ')
                plt.close()

            #Cross Correlations
            #------------------------------------------------------------------------------------

            x_corr_lst_v12 = x_corr(dv1_filt, dv2_filt, start_point_v12, end_point_v12, 'dV1', 'dV2', time, date)
            # x_corr_v12 = find_phase(x_corr_lst_v12[1], samp_rt)

            x_corr_lst_v34 = x_corr(dv3_filt, dv4_filt, start_point_v34, end_point_v34, 'dV3', 'dV4', time, date)
            # x_corr_v34 = find_phase(x_corr_lst_v34[1], samp_rt)

            if v5_inclde == True:
                x_corr_lst_v5 = x_corr(vsc, dv5_filt, start_point_v5, end_point_v5, 'Vsc', 'dV5', time, date)
                # x_corr_v5 = find_phase(x_corr_lst_v5[1], samp_rt)
                time_lag_5 = x_corr_lst_v5[0]
                time_lag_err_5 = x_corr_lst_v5[2]

            time_lag_12 = x_corr_lst_v12[0]
            time_lag_34 = x_corr_lst_v34[0]
            time_lag_err_12 = x_corr_lst_v12[2]
            time_lag_err_34 = x_corr_lst_v34[2]

            #Angle between the wave and the V1-2 boom with V1-2 and V3-4 no longer being orthogonal
            #------------------------------------------------------------------------------------
            t_12 = time_lag_12
            t_34 = time_lag_34
            t_12_err = time_lag_err_12
            t_34_err = time_lag_err_34
            angle_wave_v1 = angle_wrt_v1(t_12, t_34, wave_angle=False)
            angle = angle_wave_v1
            t_mag = t_12**2 + t_34**2
            angle_err = np.sqrt(((t_12*t_34_err)/t_mag)**2+((t_34*t_12_err)/t_mag)**2)

            #Calculating Wave Speed
            #------------------------------------------------------------------------------------
            if t_12 == 0:
                vel_12 = 0*(u.m/u.s)
                vel_12_err = 0*(u.m/u.s)
            else:
                vel_12 = abs(boom_len*np.cos(angle_wave_v1*(np.pi/180)))/t_12*(u.m/u.s)
                vel_12_err = np.sqrt((abs(angle_err*boom_len*np.sin(angle_wave_v1*(np.pi/180)))/t_12)**2 + (abs(t_12_err*boom_len*np.cos(angle_wave_v1*(np.pi/180)))/t_12**2)**2)*(u.m/u.s)
            if t_34 == 0:
                vel_34 = 0*(u.m/u.s)
                vel_34_err = 0*(u.m/u.s)
            else:
                vel_34 = abs(boom_len*np.sin(angle_wave_v1*(np.pi/180)))/t_34*(u.m/u.s)
                vel_34_err = np.sqrt((abs(angle_err*boom_len*np.cos(angle_wave_v1*(np.pi/180)))/t_34)**2 + (abs(t_34_err*boom_len*np.sin(angle_wave_v1*(np.pi/180)))/t_34**2)**2)*(u.m/u.s)
            vel_12 = vel_12.to(u.km/u.s)
            vel_34 = vel_34.to(u.km/u.s)
            vel_12_err = vel_12_err.to(u.km/u.s)
            vel_34_err = vel_34_err.to(u.km/u.s)

            #vel_tot = np.sqrt((vel_12**2)+(vel_34**2))

            #Angle between the magnetic field and the V1-2 plane
            #------------------------------------------------------------------------------------
            angle = abs(angle_wave_v1-mag_fld_angle)

            #Angle between solar wind and V_12 boom
            #------------------------------------------------------------------------------------
            sw_angle = abs(angle_wave_v1-sw_vel_angle)

            #Writing Data to txt file 
            #------------------------------------------------------------------------------------
            with open(log_file, 'a') as f: 
                f.write(f"Section : {j} ----------------------------------------------\n")
                f.write('\n')

                f.write(f"Cross Correlation section for V1-2 [s since {date[0]}]: \n")
                f.write('{} : {}'.format(start_point_v12*dt, end_point_v12*dt))
                f.write('\n'*2)

                f.write(f"Cross Correlation section for V3-4 [s since {date[0]}]: \n")
                f.write('{} : {}'.format(start_point_v34*dt, end_point_v34*dt))
                f.write('\n'*2)

                f.write("V1-2 Time Delay [s]\n")
                f.write("{} +/- {}".format(t_12, t_12_err))
                f.write('\n'*2)

                f.write("V3-4 Time Delay [s]\n")
                f.write("{} +/- {}".format(t_34, t_34_err))
                f.write('\n'*2)

                f.write("Angle between the wave and the V1 boom [Degrees]\n")
                f.write('{} +/- {}'.format(angle_wave_v1, angle_err*(180/np.pi)))
                f.write('\n'*2)

                f.write("Speed of the Wave V1 to V2 [km/s]\n")
                f.write('{} +/- {}'.format(vel_12.value, vel_12_err.value))
                f.write('\n'*2)

                f.write("Speed of the Wave V3 to V4 [km/s]\n")
                f.write('{} +/- {}'.format(vel_34.value, vel_34_err.value))
                f.write('\n'*2)

                f.write("Angle between the magnetic field and V1 Boom [Degrees]\n")
                f.write('{}'.format(mag_fld_angle))
                f.write('\n'*2)

                f.write("Angle between the wave and the magnetic field [Degrees]\n")
                f.write('{}'.format(angle))
                f.write('\n'*2)

                f.write("Angle of the magnetic field from V5 Boom [Degrees]\n")
                f.write('{}'.format(mag_angle_out))
                f.write('\n'*2)

                f.write("Magnitude of the magnetic field [nT]\n")
                f.write('{}'.format(mag_vector))
                f.write('\n'*2)

                f.write("Angle between solar wind and wave [Degrees]\n")
                f.write('{}'.format(sw_angle))
                f.write('\n'*2)

                f.write("Solar Wind Speed in X-Y Plane [km/s]\n")
                f.write('{}'.format(sw_speed.value))
                f.write('\n'*3)
            print("Would you like to time another section of this wave?")
            another_sec = ''
            while another_sec != 'y' and another_sec != 'n':
                another_sec = input('[y]/n: ')
            j = j+1
        
        print('Would you like to continue to the next trial?')
        ext = ''
        while ext != 'y' and ext != 'n':
            ext = input('[y]/n: ')
        if ext == 'n':
            exit()
