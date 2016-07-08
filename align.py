#!/usr/bin/python

import sys, os
import numpy as np

# alignment organization
import pandas as pd
import multiprocessing as mp
import h5py

# formant calculation
from numpy.lib import stride_tricks
from scipy.io.wavfile import read
from scipy.signal import filtfilt
from scipy.signal import butter
from scikits.talkbox import lpc

def swapPhones(align_file, phones_file):
    ali = pd.read_csv(align_file, sep=' ', index_col=[0], usecols=[0, 2, 3, 4], header=0, names=['fid', 'time', 'dur', 'phon'])
    ali = ali[['phon', 'time', 'dur']]
    pho = open(phones_file).read().split('\n')[:-1]

    # build dictionary of phones from phones.txt
    phonmap = {}
    for p in pho:
        val, key = p.split()
        phonmap[int(key)] = val

    # drop rows which aren't in our considered dataset
    ali.drop(ali.loc[ali.index.str.len() < 13].index, inplace=True)
    files = ali.index
    # replace phone number with actual phone
    ali.phon = ali.phon.map(phonmap)

    return ali

# this can't actually be automated cos it depends on the semantics of the data
def makeHDF(ali_file, hdf_file):
    df = pd.read_csv(ali_file, sep=' ', usecols=[0,2,3,4], header=0, names=['fid','time','dur','phon'])
    df = df.loc[df.fid.str.len() >= 13]

    spk = []
    for l in df.fid:
        spk.append(l[0]+l[6:9])
    df['sid'] = spk
    df = df[['sid', 'fid','phon','time','dur']]
    
    with h5py.File(hdf_file, 'w') as f:
        for s in set(spk):
            grp = df.loc[df.sid == s]
            g = f.create_group(s)
            for fid in grp.fid.unique():
                dset = grp.ix[grp.fid == fid, 'phon':].values
                g.create_dataset(name=fid, data=dset)

def calcPhoneFeats(df, name, all=False):
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

def getFrames(dat, frame_size):
    hop_size = int(frame_size - np.floor(0.5 * frame_size))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size/2.0))), dat)
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frame_size) / float(hop_size))) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))
    # organize frames into multidimensional matrix
    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size), strides=(samples.strides[0]*hop_size, samples.strides[0])).copy()
    return frames

def getFormants(frames, sr):
    # calculate number of LPC coefficients to use
    ncoeff = 2 + sr/1000
    # calculate LPC coefficients
    c = lpc(frames, ncoeff)[0]
    # obtain roots of LPC
    A = np.diag(np.ones((c.shape[1]-2,), float), -1)
    cs = -c[:,1:]
    Z = np.array([np.vstack((cp, A[1:])) for cp in cs])
    # root calculation using eigen method: VERY SLOW
    eig = np.linalg.eigvals(Z)
    arc = np.arctan2(np.imag(eig), np.real(eig))
    # convert to Hz and sort ascending
    formant = []
    pi2 = 0.05*sr/np.pi
    [formant.append(sorted(pi2*a[a>0])[:4]) for a in arc]

    return np.array(formant)

def calcWaveFeats(filename, fs, win, b, a):
    # read wav
    sr, raw = read(filename)
    # apply highpass filter
    raw = filtfilt(b, a, raw)
    # get raw frames
    frames = getFrames(raw, fs)
    # apply hanning window
    frames *= win
    # get energy
    energy = np.mean(np.abs(frames/2**15), axis=-1)
    # get formants
    formant = getFormants(frames, sr)
    # stack and append
    wavfeat = np.vstack((formant.T, energy.T))

    return wavfeat

def procDirMfc(args):
    s, ali, mfc, phones, pm = args
    ali = h5py.File(ali)[s]
    mfc = h5py.File(mfc)[s]
    spkr = s
    attr = ali.keys()[0][3:6]
    print(s+' '),

    # make sure files exist in both alignment and mfcc datasets
    files = list(set(ali.keys()) & set(mfc.keys()))
    # align mfcc features to phones
    mfcfeat = []
    alifeat = []
    allf = []
    sponf = []
    readf = []
    for f in files:
        # separate spon and read files
        if f[-4] == 'Z':
            sponf.append(f)
        else:
            readf.append(f)
        # read MFCC and timing/alignment data for file
        mfcdata = mfc[f].value.T
        alidata = ali[f].value
        # align MFCC and timing data
        strides = (100*alidata[:,1]).astype(int)
        end = mfcdata.shape[-1]
        mfcalign = []
        for i, s in enumerate(strides):
            e = strides[i+1] if (i+1) < strides.size else end
            mfcalign.append(np.mean(mfcdata[:,s:e], axis=-1))
            allf.append(f)
        mfcfeat.extend(np.vstack(mfcalign))
        alifeat.extend(alidata[:,[0,2]])

    mfcfeat = np.array(mfcfeat)
    numfeats = mfcfeat.shape[1]
    # delta wave features: average distance of a phone from the succeeding
    # phone
    delfeats = np.sum(np.square(np.diff(mfcfeat[:, :-1],axis=0)),axis=-1)**0.5
    delfeats = np.array(delfeats.tolist() + [0.0]).reshape((delfeats.shape[0]+1, 1))
    # horizontally stack all the data together
    feats = np.concatenate((alifeat, mfcfeat, delfeats), axis=1)
    # make dataframe from features
    dfspk = pd.DataFrame(feats, index=allf, columns=['phon','dur']+range(13)+['delfdist'])
    dfspk.phon = dfspk.phon.astype(int).map(pm)

    # calculate per phone per spontan/read stats
    dfspon = dfspk.loc[sponf]
    dfread = dfspk.loc[readf]
    dfspksz = dfspon.index.size + dfread.index.size
    phonfeats = []
    phones = list(set(dfspon.phon) & set(dfread.phon) & set(phones))
    for p in phones:
        # gather per phone data for spon/read: leave out phone name and time
        dfsponp = dfspon.ix[dfspon.phon == p, 2:]
        dfreadp = dfread.ix[dfread.phon == p, 2:]
        # calculate phone frequency of occurence for spon/read
        sfreq = float(dfsponp.index.size)/dfspksz
        rfreq = float(dfreadp.index.size)/dfspksz
        # calculate per phone means for spon/read, and stack
        smean = dfsponp.mean().values
        rmean = dfreadp.mean().values
        diffp = smean - rmean
        # finally, calculate differences between spon/read features
        # from left to right: duration, wave features, energy, delta features,
        # delta energy
        deldur = diffp[0]/(smean[0] + rmean[0])
        # for wave features (formant/mfcc), calculate euclidian distance
        # between spon and read means
        delf = np.sum(np.square(diffp[1:numfeats]))**0.5
        deldelf = diffp[-1]
        # for energy, subtract for mfcc (because it's log), calculate decibel
        # if formant
        deleng = diffp[numfeats]
        # append results
        phonfeats.append([spkr, attr[0], attr[1], attr[2],
            sfreq, rfreq, deldur, delf, deleng, deldelf])

    try:
        phonfeats = pd.DataFrame(phonfeats, index=phones,
                columns=['spkr', 'k', 's', 'u', 'sfreq', 'rfreq',
                    'dur', 'fdist', 'energy', 'delfdist'])
    except:
        print('no phone feats! ' + spkr)
        return
    # return processed features
    return phonfeats

def procDir(args):
    pathin, pathout, df, phones, features, overwrite = args

    # get frame size
    sr = 16000        # samplerate
    fss = 0.01        # framesize in seconds
    hpf = 50.         # highpass freq in Hz
    fs = int(fss*sr)  # framesize in samples
    # calc hanning window
    win = np.hanning(fs)
    # calc highpass filter coefficients
    b, a = butter(1, 50./(0.5*sr), "highpass")
    # get wav files in specified directory
    wavsin = []
    argout = []
    for dirpath,_,filename in os.walk(pathin):
        wavsin.extend([os.path.join(dirpath,f) for f in filename if f.split('.')[-1] == 'wav'])
        argout.extend([f.replace('.wav','') for f in filename if f.split('.')[-1] == 'wav'])
    paths = dict(zip(argout, wavsin))

    # filter both ways to ensure only files that exist on both dataframes are
    # processed
    try:
        dfspk = df.loc[argout].dropna()
    except:
        return
    files = dfspk.index.unique()

    # if we've calculated the wave feats before, load the file containing
    # the features dictionary
    saved = False
    fileout = os.path.join(pathout, os.path.split(pathin)[-1]) + '.npz'
    if not overwrite:
        saved = True
        if features == 'formant':
            if os.path.exists(fileout):
                featdict = np.load(fileout)
            else:
                saved = False
                wavfeats = {}
        elif features == 'mfcc':
            featdict = np.load('feats-inti.npz')
    else:
        wavfeats = {}

    feats = []
    for n, f in enumerate(files):
        path = paths[f]
        if not os.path.exists(path): continue
        # calc wave feats: f0-f4 + energy: this is VERY SLOW
        if not saved and features == 'formant':
            wavfeat = calcWaveFeats(path, fs, win, b, a)
            wavfeats[f] = wavfeat
        else:
            # if already calculated, load from dict
            wavfeat = featdict[f]
            if features == 'mfcc': wavfeat = wavfeat[::-1]
        # align formant to phone data
        timing = dfspk.loc[f].time
        foffset = 200. if features == 'formant' else 100.
        strides = (foffset*(timing.get_values())).astype(int)
        end = wavfeat.shape[-1]
        falign = []
        for i, s in enumerate(strides):
            e = strides[i+1] if (i+1) < strides.size else end
            falign.append(np.mean(wavfeat[:,s:e], axis=-1))
        falign = np.vstack(falign)
        feats.extend(np.vstack(falign))

    if not saved:
        np.savez_compressed(fileout.replace('.npz',''), **wavfeats)

    feats = np.array(feats)
    numfeats = feats.shape[1]
    # delta wave features: average distance of a phone from the succeeding
    # phone
    delfeats = np.sum(np.square(np.diff(feats[:, :-1],axis=0)),axis=-1)**0.5
    delfeats = np.array(delfeats.tolist() + [0.0]).reshape((delfeats.shape[0]+1, 1))
    feats = np.concatenate((feats, delfeats), axis=1)
    newcols = [str(n) for n in range(numfeats+1)]
    for n in range(len(newcols)):
        dfspk[newcols[n]] = feats[:,n]

    # calculate per phone per spontan/read stats
    dfspon = dfspk.loc[[f for f in files if f.find('Z') > 0]]
    dfread = dfspk.loc[[f for f in files if f.find('Z') < 0]]
    dfsponsz = dfspon.index.size
    dfreadsz = dfread.index.size
    phonfeats = []
    phones = [p for p in phones if p in dfspon.phon.unique() and p in dfread.phon.unique()]
    for p in phones:
        # gather per phone data for spon/read: leave out phone name and time
        dfsponp = dfspon.ix[dfspon.phon == p, 2:]
        dfreadp = dfread.ix[dfread.phon == p, 2:]
        # calculate phone frequency of occurence for spon/read
        ns = float(dfsponp.index.size)/dfsponsz
        nr = float(dfreadp.index.size)/dfreadsz
        # calculate per phone means for spon/read, and stack
        phonfeat = np.vstack((dfsponp.mean().get_values(), dfreadp.mean().get_values()))

        # finally, calculate differences between spon/read features
        # from left to right: duration, wave features, energy, delta features,
        # delta energy
        deldur = phonfeat[0,0]-phonfeat[1,0]
        # for wave features (formant/mfcc), calculate euclidian distance
        # between spon and read means
        delf = np.sum(np.square(np.diff(phonfeat[:, 1:numfeats],axis=0)))**0.5
        deldelf = phonfeat[0,-1]-phonfeat[1,-1]
        # for energy, subtract for mfcc (because it's log), calculate decibel
        # if formant
        if features == 'mfcc':
            deleng = phonfeat[0,numfeats]-phonfeat[1,numfeats]
        elif features == 'formant':
            deleng = 20.*np.log10(phonfeat[0,numfeats]/phonfeat[1,numfeats])
        # append results
        phonfeats.append([ns, nr, deldur, delf, deleng, deldelf ])

    # return processed features
    #return pd.DataFrame(phonfeats, index=phones, columns=['ns', 'nr', 'dur', 'fdist', 'energy', 'delfdist'])
    return np.vstack(phonfeats)

def calcFeats(alifile, featfile, phonfile):
    try:
        ali = h5py.File(alifile)
    except:
        print('alignment file must be in HDF5 format.')
        sys.exit()

    try:
        mfc = h5py.File(featfile)
    except:
        print('MFCC features file must be in HDF5 format.')
        sys.exit()

    phones = [p for p in open(phonfile).read().split('\n')[:-1]]
    phonemap = {}
    for line in open('phones.txt').read().split('\n')[:-1]:
        p, n = line.split()
        phonemap[int(n)] = p

    # make sure speaker exists both in alignment and MFCC files
    spk = list(set(ali.keys()) & set(mfc.keys()))
    ali.close()
    mfc.close()
    args = [[s, alifile, featfile, phones, phonemap] for s in spk]

    pool = mp.Pool(mp.cpu_count())
    feats = pool.map(procDirMfc, args)
    pool.close()
    pool.terminate()
    pool.join()

    return feats

def viewPerPhone(dfp, attr=None):
    phones = [p for p in dfp.index.unique() if dfp.loc[p].values.ndim > 1]
    if attr:
        means = [np.mean(dfp.loc[p].ix[lambda df: df[attr[0]] == attr[1], 4:].values, axis=0) for p in phones]
        dfpp = pd.DataFrame(means, index=phones, columns=dfp.columns[4:])
    else:
        dfpp = pd.DataFrame([np.mean(dfp.ix[p, 4:].values, axis=0) for p in phones], index=phones, columns=dfp.columns[4:])
    return dfpp

def main(args):
    confdir = args[1]
    formdir = args[3]
    wavsdir = args[2]

    # check if formants have been calculated
    if not os.path.exists(formdir):
        print('calculating formants from wavs')
        calcWaveFeats(wavsdir, formdir)

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

