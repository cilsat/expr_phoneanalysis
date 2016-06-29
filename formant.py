import sys
import os
import numpy as np
import multiprocessing as mp
from numpy.lib import stride_tricks
from scipy.io.wavfile import read
from scipy.signal import filtfilt
from scipy.signal import butter
from scikits.talkbox import lpc

def getFrames(args):
    filename = args[0]
    frame_size = args[1]
    window = args[2]
    filt = args[3]

    sr, dat = read(filename)
    hop_size = int(frame_size - np.floor(0.5 * frame_size))

    # high pass filter input signal
    dat = filtfilt(filt[0], filt[1], dat)
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size/2.0))), dat)
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frame_size) / float(hop_size))) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    # organize frames into multidimensional matrix
    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size), strides=(samples.strides[0]*hop_size, samples.strides[0])).copy()
    # apply hanning window
    frames *= window

    # calculate number of LPC coefficients to use
    ncoeff = 2 + sr/1000
    # calculate LPC coefficients
    c = lpc(frames, ncoeff)[0]
    # obtain roots of LPC
    pi2 = 0.5*sr/np.pi
    formant = []
    """
    for co in c:
        roots = np.roots(co)
        roots = [r for r in roots if np.imag(r) > 0]
        ang = np.arctan2(np.imag(roots), np.real(roots))
        formant.append(sorted(ang*pi2)[:4])
    """
    A = np.diag(np.ones((c.shape[1]-2,), float), -1)
    cs = -c[:,1:]
    Z = np.array([np.vstack((cp, A[1:])) for cp in cs])
    eig = np.linalg.eigvals(Z)
    arc = np.arctan2(np.imag(eig), np.real(eig))
    [formant.append(sorted(pi2*a[a>0])[:4]) for a in arc]

    return np.array(formant)

def procFolder(args):
    pathin = args[0]
    pathout = args[1]

    ms = 0.01
    sr = 16000
    # get frame size
    fs = int(ms*sr)
    # calc hanning window
    win = np.hanning(fs)
    # calc highpass filter coefficients
    b, a = butter(1, 50./(0.5*sr), "highpass")
    # get wav files in specified directory
    if not os.path.exists(pathout):
        os.mkdir(pathout)
    wavsin = []
    fout = []
    for f in os.listdir(pathin):
        if f.split('.')[-1] == "wav":
            wavsin.append(os.path.join(pathin, f))
            out = f.replace("wav", "f")
            fout.append(os.path.join(pathout, out))

    for n, wav in enumerate(wavsin):
        formants = getFrames([wav, fs, win, [b,a]])
        with open(fout[n], 'wb') as f:
            [f.write(','.join(str(ff) for ff in form.tolist())+'\n') for form in formants]

if __name__ == "__main__":
    path = sys.argv[1]
    pool = mp.Pool(4*mp.cpu_count())
    pool.map(procFolder, os.listdir(path))
