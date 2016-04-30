__author__ = 'Air'

import numpy as np
import scipy.signal as sg
import scipy.fftpack as scifft
import wave
from math import log

'''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
#conv & corr
def convm(x, p):
    N = len(x) + 2*p - 2
    xpad = np.concatenate([np.zeros(p-1), x[:], np.zeros(p-1)])
    X = np.zeros((len(x)+p-1, p))
    for i in xrange(p):
        X[:, i] = xpad[p-i-1:N-i]
    return X


def conv_full(x):
    x_ = x[:]
    p = len(x_)
    m = len(x_)
    convar_ = corr_full(x)
    averge_array = np.ones(p)*np.average(x_)
    result_ = convar_ - averge_array*averge_array.T
    return result_

def conv(x, p):
    x = x[:]
    m = len(x)
    x = x - np.ones(m) * np.sum(x) / m
    result_ = np.dot(convm(x, p).conj().T, convm(x, p)) / (m-1)
    return result_

def autocorrelation(x):
    x_ = np.array(x, dtype=np.float64)
    corr = np.correlate(x_, x_, 'full')
    corr = corr[len(corr)-len(x_):]
    for i in range(0, len(corr)):
        corr[i] /= (len(x_)-i)
    return corr


def corr_full(x):
    x_ = np.array(x, dtype=np.float64)
    x_ = autocorrelation(x_)
    result_ = np.zeros((len(x_), len(x_)))
    xpad = np.concatenate([x_[len(x_):0:-1], x_])
    for i in range(0, len(x_)):
        result_[i, :] = xpad[len(xpad)-len(x_)-i:len(xpad)-i]
    return result_


def corr(x, p):
    x = x[:]
    m = len(x)
    result_ = np.dot(convm(x, p).conj().T, convm(x, p)) / (m-1)
    return result_

'''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
#signal model methonds
def prony_method(x, p, q):
    N = len(x)
    X = convm(x, p+1)
    Xq = X[q:N+p-1, 0:p]
    Xq1 = -X[q+1:N+p, 0]

    a = np.linalg.lstsq(Xq, Xq1)[0]
    a = np.insert(a, 0, 1)
    b = np.dot(X[0:q+1, 0:p+1], a)

    return b, a


'''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
#power Spectrum Estimation
def periodigram(x, p, q):
    handle = x[p:q]
    Px = np.abs(np.fft.fft(handle)*2/(q-p+1))
    freq = np.fft.fftfreq(len(handle))
    Px[1] = Px[2]
    return freq, Px


def ARMA(x, p, q):
    x_ = x[:]
    b_, a_ = prony_method(x_ , p, q)
    w, h = sg.freqz(b_, a_, len(x_))
    return w, h

'''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
#witeing the noise
def noiseWhite(x):
    x_ = x[:]
    datalen = len(x_)
    conv_ = conv_full(x_)
    w, v = np.linalg.eig(conv_)

    T = v.dot(np.linalg.inv(np.diag(w)))
    whiteNoise = T.dot(x_)
    return whiteNoise


def complex_pow(x, y):
    while y > 0:
        x *= x
        y -= 1
    return x

def frequency_estimate_ls(x, len_):
    '''
    x_ = scifft.hilbert(x)
    x_ = noiseWhite(x_)
    datalen = len(x_)
    estaminlen = 10
    pool = np.zeros(datalen*estaminlen, "complex64").reshape(datalen, estaminlen)
    for i in range(0, datalen):
        for j in range(0, estaminlen):
            pool[i, j] = complex_pow(0+1j, j)*np.math.pow(i, j)/np.math.factorial(j)
    a = np.linalg.lstsq(pool, x_)[0]
    result_ = abs(a[0])
    return result_
    '''
    x_ = x[0:len_]
    #x_ = noiseWhite(x_)
    datalen = len(x_)
    estaminlen = 10
    pool = np.zeros(datalen*estaminlen).reshape(datalen, estaminlen)
    for i in range(0, datalen):
        for j in range(0, estaminlen):
            stage = 2*j+1
            pool[i, j] = np.math.pow(-1, j+2)*np.math.pow(i, stage)/np.math.factorial(stage)
    #pool *= 10000
    #x_ *= 1000000
    a = np.linalg.lstsq(pool, x_)[0]
    result_ = a[0]
    return result_

def frequency_estimate_svd(x, p):
    #x_ = scifft.hilbert(x)
    x_ = x[:]
    #x_ = noiseWhite(x_)
    datalen = len(x_)
    x_ = corr(x_, p)
    w, v = np.linalg.eig(x_)
    Index = np.argmax(w)
    Px = np.zeros(p)
    for i in range(0, p):
        if i != Index:
            Px += abs(np.fft.fft(v[i, :]))
    #Px = -20*np.log10(Px)
    #vec = v[np.argmax(w)]
    #vec_fft = np.fft.fft(vec)
    #fft_max = np.argmax(vec_fft)
    freqP = np.fft.fftfreq(len(v[0, :]))
    fft_max = freqP[np.argmax(Px)]
    return fft_max*10


def frequency_estimate_ml(x):
    x_ = scifft.hilbert(x)
    x_ = noiseWhite(x_)
    datalen = len(x_)

    for i in range(0, datalen):
        x_[i] = np.math.log(abs(x_[i]))/(i+1)
    averge_ = np.average(x_)
    result_ = abs(averge_)
    return result_


#notation extractor
def musicNotation(fileName):
    #open a wave file, and return a Wave_read object
    f = wave.open(fileName, "rb")
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

    framePerSecond = 1.0/framerate
    #every notaion we should take such number frames
    peace = 93.0
    framePerNotaion = int(np.floor((60.0/peace)/framePerSecond))

    realeased_notations = []
    note_list = ['0','0','1','2','2','3','3','4','4','5','6','6']
    for i in range(0, nframes, framePerNotaion):
        symbol = wave_data[0,i:i+framePerNotaion]
        symbolFFT = np.fft.fft(symbol)
        symbolFFT = abs(symbolFFT)
        Index = np.argmax(symbolFFT)
        freq = np.fft.fftfreq(len(symbolFFT))
        freMax = abs(freq[Index])
        if freMax > 0:
            #interval = int(round(log(freMax/440.0, 1.059463))) % 12
            interval = int(round(log((freMax*framerate)/440.0, 1.059463))) % 12
            realeased_notations.append(note_list[interval])
        #else:
        #    realeased_notations.append('7')
    return realeased_notations,framerate

def genMusic(realeased_notations,framerate):
    #waveBase = np.zeros(7*54400)
    #waveBase.shape = 7, -1
    startPoint = 600
    endPoint = 30000
    waveBase = []
    for i in range(1,8):
        name = str(i) + '.wav'
        f = wave.open(name, "rb")
        params = f.getparams()
        _1, _2, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()
        basicWave = np.fromstring(str_data, dtype = np.short)
        basicWave.shape = -1, 2
        #transpose the data
        basicWave = basicWave.T
        basicWave = basicWave[0, startPoint:endPoint]
        #waveBase[i-1,:]=basicWave
        waveBase.append(basicWave.T)

    #waveBase.append(list(np.zeros(len(waveBase[0]))))

    music = []
    for i in realeased_notations:
        mvox = list(waveBase[int(i)][:])
        music = music + mvox

    music = np.array(music)

    f = wave.open(r"out.wav", "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(framerate)
    f.writeframes(music.tostring())
    f.close()



