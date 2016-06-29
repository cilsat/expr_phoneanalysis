#!/usr/bin/python

import sys, os
import numpy as np

# alignment organization
import pandas as pd
import multiprocessing as mp

# formant calculation
from numpy.lib import stride_tricks
from scipy.io.wavfile import read
from scipy.signal import filtfilt
from scipy.signal import butter
from scikits.talkbox import lpc

def swapPhones(align_file, phones_file):
    ali = open(align_file).read().split('\n')[:-1]
    pho = open(phones_file).read().split('\n')[:-1]

    # build dictionary of phones from phones.txt
    phones = {}
    for p in pho:
        val, key = p.split()
        phones[key] = val

    # replace phone number with actual phone
    spon = []
    read = []
    for i, a in enumerate(ali):
        s = a.split()
        p = phones[s[-1]]
        if len(s[0]) == 13:
            if s[0].find('Z') > 0:
                spon.append([s[0], float(s[2]), float(s[3]), p])
            else:
                read.append([s[0], float(s[2]), float(s[3]), p])

    # make pandas dataframe of filename, location, duration, and name of phone
    spon = pd.DataFrame(spon)
    read = pd.DataFrame(read)
    spon.rename(columns={0:'fil', 1:'loc', 2:'dur', 3:'phon'}, inplace=True)
    read.rename(columns={0:'fil', 1:'loc', 2:'dur', 3:'phon'}, inplace=True)
    spon = spon[['fil', 'phon', 'loc', 'dur']]
    read = read[['fil', 'phon', 'loc', 'dur']]
    return spon, read

def calcFeats(df, name, all=False):
    #get vowels
    if all:
        phones = [p for p in df['phon'].unique()]
    else:
        phones = []
        for p in df['phon'].unique():
            v = p.split('_')[0]
            if v.startswith('a') or v.startswith('e') or v.startswith('i') or v.startswith('o') or v.startswith('u') or v.startswith('@'):
                phones.append(p)

    # calcculate delta formants
    delf = df.iloc[:,4:].diff()
    delf.iloc[0,:] = 0
    delf.rename(columns={'f0':'df0', 'f1':'df1', 'f2':'df2', 'f3':'df3'}, inplace=True)
    df[delf.columns] = delf

    # calculate the average/variance of freq, duration, formants 0-4 of each vowel
    dfsize = 1./df.size
    feats = []
    for p in phones:
        phon = df[df['phon'] == p]
        num = len(phon.index)
        fr = num*dfsize*100
        dm = phon['dur'].mean()
        dv = phon['dur'].var()
        fm = phon.loc[:,'f0':'f3'].mean().tolist()
        fv = phon.loc[:,'f0':'f3'].var().tolist()
        dfm = phon.loc[:,'df0':'df3'].abs().mean().tolist()
        dfv = phon.loc[:,'df0':'df3'].abs().var().tolist()
        feats.append([fr, num, dm, dv]+fm+fv+dfm+dfv)
    cols = [('occurence', name+'freq'), ('occurence', name+'num'), ('dur', name+'mean'), ('dur', name+'var')]
    cols += [('formants', name+str(n)+'mean') for n in range(4)]
    cols += [('formants', name+str(n)+'var') for n in range(4)]
    cols += [('deltas', name+str(n)+'mean') for n in range(4)]
    cols += [('deltas', name+str(n)+'var') for n in range(4)]
    mcols = pd.MultiIndex.from_tuples(cols)

    return pd.DataFrame(feats, index=phones, columns=mcols)

def calcSR(spon, read, all=False):
    sfeats = calcFeats(spon, 's', all)
    rfeats = calcFeats(read, 'r', all)
    feats = pd.concat((sfeats, rfeats), axis=1)
    return feats

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
    wavsin = []
    fout = []
    for f in os.listdir(pathin):
        if f.split('.')[-1] == "wav":
            wavsin.append(os.path.join(pathin, f))
            out = f.replace("wav", "f")
            fout.append(os.path.join(pathout, out))

    for n, wav in enumerate(wavsin):
        if not os.path.exists(wav): continue
        formants = getFrames([wav, fs, win, [b,a]])
        with open(fout[n], 'wb') as f:
            [f.write(','.join(str(ff) for ff in form.tolist())+'\n') for form in formants]

def calcFormants(wavsdir, formdir):
    if not os.path.exists(formdir):
        os.mkdir(formdir)
    pool = mp.Pool(mp.cpu_count())
    args = [[d, formdir] for d in wavsdir if os.path.exists(d)]
    pool.map(procFolder, args)
    pool.close()
    pool.terminate()
    pool.join()

def procFile(args):
    split = args[0]
    df = args[1]
    formdir = args[2]

    ff = []
    for f in split:
        timing = df[df['fil'] == f]
        formants = []
        for row in open(os.path.join(formdir, f+'.f')).read().split('\n')[:-1]:
            formants.append([float(num) for num in row.split(',')])
        formants = np.array(formants)
        end = formants.shape[0]
        strides = (timing['loc']*200).get_values().astype(int)
        for i, s in enumerate(strides):
            e = strides[i+1] if (i+1) < strides.size else end
            ff.append(np.mean(formants[s:e,:].T,axis=-1))
    return np.vstack(ff)

def swapFormants(df, formdir):
    files = df['fil'].unique()
    for f in files:
        if not os.path.exists(os.path.join(formdir, f+'.f')):
            print('formant file ' + f + ' does not exist!')
            sys.exit(1)

    split = []
    nsplits = mp.cpu_count()
    s = files.size/nsplits
    for n in range(nsplits):
        e = (n+1)*s if (n+1) < nsplits else files.size
        split.append(files[n*s:e])

    pool = mp.Pool(mp.cpu_count())
    formfeats = pool.map(procFile, [[f, df, formdir] for f in split])
    pool.close()
    pool.terminate()
    pool.join()
    formfeats = np.vstack(formfeats)
    print(formfeats)
    print(formfeats.shape)
    df['f0'] = formfeats[:,0]
    df['f1'] = formfeats[:,1]
    df['f2'] = formfeats[:,2]
    df['f3'] = formfeats[:,3]

def main(args):
    confdir = args[1]
    formdir = args[3]
    wavsdir = args[2]

    # check if formants have been calculated
    if not os.path.exists(formdir):
        print('calculating formants from wavs')
        calcFormants(wavsdir, formdir)

    align_file = os.path.join(confdir, 'ali.ctm')
    phones_file = os.path.join(confdir, 'phones.txt')

    print('substituting phone names')
    spon, read = swapPhones(align_file, phones_file)
    print('aligning spontan formants')
    swapFormants(spon, formdir)
    print('aligning read formants')
    swapFormants(read, formdir)
    print('calculating per phone features')
    feats = calcSR(spon, read)
    print(feats)

if __name__ == "__main__":
    main(sys.argv)

