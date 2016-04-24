# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:09:32 2016

@author: lianying
"""

import numpy as np
from scipy.linalg import eigvals, pinv, svd, eig
from scipy.fftpack import fft, rfft, irfft
import scipy.signal as sg
import matplotlib.pyplot as plt
import math
import wave
import mathtools
#from pylab import *


# generate gaussian white noise;
def gen_white_noise(n):
    return np.array([np.random.randn() for i in xrange(n)])


# generate sampled sinusoid signal
def gen_sin_sig(amp, f, fs, tau):
    nT = np.linspace(0, tau, round(tau/(1.0/fs)))
    signal =np.array([amp*np.sin(2*np.pi*f/fs*t) for t in nT])
    return signal


# analyze H(z) of a IIR filter
def plot_hz(b,a):
    w, h = sg.freqz(b, a)
    #h_dB = 20 * np.log10(abs(h))
    plt.figure()
    #plt.plot(w/max(w), h_dB)
    plt.plot(w/max(w), h)
    return h

def plot_hn(b, a=1):
    l = len(b)*10
    impulse = np.repeat(0.0,l)
    impulse[0] =1.0
    x = np.arange(0,l)
    response = sg.lfilter(b, a, impulse)
    plt.figure()
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')
    return response


def read_wave_data(file_path):
  #open a wave file, and return a Wave_read object
  f = wave.open(file_path, "rb")
  #read the wave's format infomation,and return a tuple
  params = f.getparams()
  #get the info
  nchannels, sampwidth, framerate, nframes = params[:4]
  #Reads and returns nframes of audio, as a string of bytes. 
  str_data = f.readframes(nframes)
  #close the stream
  f.close()
  #turn the wave's data to array
  wave_data = np.fromstring(str_data, dtype = np.short)
  #for the data is stereo,and format is LRLRLR...
  #shape the array to n*2(-1 means fit the y coordinate)
  wave_data.shape = -1, 2
  #transpose the data
  wave_data = wave_data.T
  #calculate the time bar
  time = np.arange(0, nframes) * (1.0/framerate)
  return wave_data, time

# design filter using signal.iirdesign
b, a = sg.iirfilter(2, 0.2, rs=30,btype='highpass', ftype='cheby2')
hz = plot_hz(b, a)
hn = plot_hn(b, a)
print "b,a of filter are", b, a


# the signal i will show you
st = gen_sin_sig(1, 2.24, 10.0, 100.0)
#st = gen_sin_sig(100, 10000.0, 10.0, 100.0)
wnt = gen_white_noise(len(st))
cnt = sg.lfilter(b,a,wnt)
wxt = st+wnt
cxt = st+cnt

plt.figure()
plt.subplot(311)
plt.plot(st)
plt.subplot(312)
plt.plot(cnt)
plt.subplot(313)
plt.plot(cxt)
plt.title("signal and nosie and noisy signal")

# get a wave file
wave_data, time = read_wave_data("2016-03-31_09_52_09.wav")	
plt.figure()
plt.subplot(211)
plt.plot(time, wave_data[0])
plt.subplot(212)
plt.plot(time, wave_data[1], c = "g")
#plt.show()

# :) now it is your turn
#    note that b,a, f is unknown
#-----------------------------------------
# STEP 1:
# TO DO: using prony method to model hn
#-----------------------------------------
b_, a_ = mathtools.prony_method(hn, 2, 2)
print "prony method result:b=", b_, "    a=", a_


#------------------------------------------------------
# STEP 2:
# TO DO: Estimate PSD of cnt using periodogram
#-----------------------------------------------------
f_, c_ = mathtools.periodigram(cnt, 1, len(cnt))


#-----------------------------------------
# STEP 3:
# TO DO: Estimate PSD of cnt using ARMA
#-----------------------------------------
w_, d_ = mathtools.ARMA(cnt, 2, 2)


plt.figure()
plt.subplot(311)
plt.plot(cnt)
plt.subplot(312)
plt.plot(f_, c_)
plt.subplot(313)
plt.plot(w_, d_)

plt.show()


#-----------------------------------------
# STEP 4:
# TO DO: Whitening cnt back to wnt
#-----------------------------------------
white_cnt = mathtools.noiseWhite(cnt)

plt.figure()
plt.subplot(211)
plt.plot(white_cnt)
plt.title("white noise")
plt.subplot(212)
plt.plot(cnt)
plt.title("color noise")
#plt.show()

#-----------------------------------------
# STEP 5:
# TO DO: Estimate frequency of sin cxt using LS
#-----------------------------------------
freq_ls = mathtools.frequency_estimate_ls(st, 3)

#-----------------------------------------
# STEP 6:
# TO DO: Estimate frequency of sin from cxt using SVD
#-----------------------------------------
#x = mathtools.autocorrelationMatrix(cnt_)
freq_svd = mathtools.frequency_estimate_svd(cxt, len(cxt)/5)
#print

#-----------------------------------------
# STEP 7:
# TO DO: Estimate frequency of sin from wxt using ML
#-----------------------------------------
#assert we know the amp and frez
freq_ml = mathtools.frequency_estimate_ml(wxt)
print "ml", freq_ml, "svd", freq_svd, "ls", freq_ls


#-----------------------------------------
# STEP 8:
# TO DO: analyze the wave signal, extract its notes
#-----------------------------------------



#-----------------------------------------
# STEP 9:
# TO DO: reform it into piano
#-----------------------------------------

