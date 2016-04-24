__author__ = 'Air'
import numpy as np
import mathtools as mtt
import matplotlib.pyplot as plt
import scipy.signal as sg
import wave
import numpy as np
import matplotlib.pyplot as plt
'''
fw = wave.open('2016-03-31_09_52_09.wav','r')
soundInfo = fw.readframes(-1)
soundInfo = np.fromstring(soundInfo,np.int16)
f = fw.getframerate()
fw.close()

plt.subplot(211)
plt.plot(soundInfo)
plt.ylabel('Amplitude')
plt.title('Wave from and spectrogram of aeiou.wav')

plt.subplot(212)
plt.specgram(soundInfo,Fs = f, scale_by_freq = True, sides = 'default')
plt.ylabel('Frequency')
plt.xlabel('time(seconds)')
plt.show()
'''
'''
nT = np.arange(1000)
signal = np.array([np.sin(0.3*t) for t in nT])
freq = np.fft.fftfreq(nT.shape[-1])
b, a = sg.iirfilter(2, 0.2, rs=30,btype='lowpass', ftype='cheby2')
white_noise = np.array([np.random.randn() for i in nT])*4
color_noise = sg.lfilter(b,a,white_noise)
color_signal = signal+color_noise
white_signal = signal+white_noise
#signal_freq = np.fft.fftfreq(nT.shape[-1])
signal_fourier = np.fft.fft(signal)
#fourier = abs(fourier)
color_signal_fourier = np.fft.fft(color_signal)
#fourier2 = abs(fourier2)
plt.figure(1)
plt.subplot(311)
plt.plot(color_signal)
plt.subplot(312)
plt.plot(freq,abs(signal_fourier))
plt.subplot(313)
plt.plot(freq, abs(color_signal_fourier))
plt.show()
'''

'''
noise = np.ones(len(nT))*2

x+=noise

plt.figure(1)
plt.subplot(311)
plt.plot(nT,x)

a_ = mtt.corr(x)

c_ = mtt.conv(x, len(x))
#print x_
#print a_
#print b_
plt.subplot(312)
plt.plot(nT, a_[1,:])
plt.subplot(313)
plt.plot(nT ,c_[1,:])
plt.show()
'''

t = np.arange(500)
sp = np.fft.fft(np.sin(0.1*t))
freq = np.fft.fftfreq(t.shape[-1])
plt.figure(1)
plt.subplot(211)
plt.plot(freq, sp.real, freq, sp.imag)
plt.subplot(212)
plt.plot(np.sin(0.1*t))
plt.show()

'''
datalen = 4
estaminlen = 4
pool = np.zeros(datalen*estaminlen).reshape(datalen,estaminlen)
for i in range(0, datalen):
    for j in range(0, estaminlen):
        stage = 2*j+1
        pool[i, j] = np.math.pow(-1, j+2)*np.math.pow(i, stage)/np.math.factorial(stage)

coff = pool[1, :]
aaa = np.ones(estaminlen)
res = coff.dot(aaa)
print res
bbb = sum(coff)
print pool
for i in range(0, datalen):
    ss = sum(pool[i, :])
    print ss
'''