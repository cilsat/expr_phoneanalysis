#!/usr/bin/python

from __future__ import print_function
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

def procAliMfc(args):
    spkr, ali, mfc, phones, pm, perphone = args

    def procSponRead(dfspk, phones):
        spkr = dfspk.index[0]
        attr = spkr[3:6]
        spkr = spkr[0] + spkr[6:9]
        # calculate per phone per spontan/read stats
        dfspon = dfspk.loc[dfspk.index.str.get(-4) == 'Z']
        dfread = dfspk.loc[set(dfspk.index) - set(dfspon.index)]
        phonfeats = []
        phones = list(set(dfspon.phon) & set(dfread.phon) & set(phones))
        for p in phones:
            # gather per phone data for spon/read: leave out phone name and time
            dfsponp = dfspon.ix[dfspon.phon == p, 'dur':]
            dfreadp = dfread.ix[dfread.phon == p, 'dur':]
            # calculate phone frequency of occurence for spon/read
            snum = dfsponp.index.size
            rnum = dfreadp.index.size
            sfreq = float(dfsponp.index.size)/dfspon.index.size
            rfreq = float(dfreadp.index.size)/dfread.index.size
            #freq = 10*np.log10(sfreq/rfreq)
            # calculate per phone means for spon/read, and stack
            smean = np.mean(dfsponp.values, axis=0).tolist()
            rmean = np.mean(dfreadp.values, axis=0).tolist()

            phonfeats.append([spkr, attr[0], attr[1], attr[2],
                snum, rnum, sfreq, rfreq] + smean + rmean)

        try:
            phonfeats = pd.DataFrame(phonfeats, index=phones)
        except:
            print('no phone feats! ' + spkr)
            return
        # return processed features
        return phonfeats

    def alignPhones(data, stride):
        align = []
        for i, s in enumerate(stride):
            e = stride[i+1] if (i+1) < stride.size else data.shape[-1]
            align.append(np.mean(data[:,s:e], axis=-1))
        return np.array(align)

    ali = h5py.File(ali)[spkr]
    mfc = h5py.File(mfc)[spkr]
    # make sure files exist in both alignment and mfcc datasets
    files = list(set(ali.keys()) & set(mfc.keys()))
    # align mfcc features to phones
    mfcfeat = []
    alifeat = []
    allf = []
    for f in files:
        # read MFCC and timing/alignment data for file
        mfcdata = mfc[f].value.T
        alidata = ali[f].value
        # align MFCC and timing data
        strides = (100*alidata[:,1]).astype(int)
        mfcfeat.extend(alignPhones(mfcdata, strides))
        alifeat.extend(alidata[:,[0,2]])
        allf.extend([f]*len(strides))

    mfcfeat = np.array(mfcfeat)
    numfeats = mfcfeat.shape[1]
    # delta wave features: average distance of a phone from the succeeding
    #ephone
    delfeats = np.sum(np.square(np.diff(mfcfeat[:, :-1],axis=0)),axis=-1)**0.5
    delfeats = np.array(delfeats.tolist() + [0.0]).reshape((delfeats.shape[0]+1, 1))
    # horizontally stack all the data together
    feats = np.concatenate((alifeat, mfcfeat, delfeats), axis=1)
    # make dataframe from features
    dfspk = pd.DataFrame(feats, index=allf, columns=['phon','dur']+range(13)+['delfdist'])

    if perphone:
        # map phones from integer to string
        dfspk.phon = dfspk.phon.astype(int).map(pm)
        dfspk = procSponRead(dfspk, phones)
    else:
        # generate labels
        dfspk['lbl'] = dfspk.index.map(lambda x: True if x[-4] == 'Z' else False)

    sys.stdout.flush()
    print(spkr, end=' ')

    return dfspk


def calcFeats(alifile, mfcfile, phonfile, perphone=True):
    try:
        ali = h5py.File(alifile)
        mfc = h5py.File(mfcfile)
    except:
        print('files must be in HDF5 format.')
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
    args = [[s, alifile, mfcfile, phones, phonemap, perphone] for s in spk]

    pool = mp.Pool(mp.cpu_count())
    feats = pool.map(procAliMfc, args)
    pool.close()
    pool.terminate()
    pool.join()

    df = pd.concat(feats)
    cols = dict(zip(range(len(df.columns)), ['spkr', 'k', 's', 'u', 'snum', 'rnum', 'sfreq', 'rfreq', 'sdur', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 'sener', 'sdist', 'rdur', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'rener', 'rdist']))
    if perphone:
        try:
            df.rename(columns=cols, inplace=True)
        except:
            print("could not rename columns")

    return df

def viewPerPhone(dfp, attr=None):
    phones = [p for p in dfp.index.unique() if dfp.loc[p].values.ndim > 1]
    if attr:
        means = [np.mean(dfp.loc[p].ix[lambda df: df[attr[0]] == attr[1], 4:].values, axis=0) for p in phones]
        dfpp = pd.DataFrame(means, index=phones, columns=dfp.columns[4:])
    else:
        dfpp = pd.DataFrame([np.mean(dfp.ix[p, 4:].values, axis=0) for p in phones], index=phones, columns=dfp.columns[4:])
    return dfpp

def predictSponRead(attrs):
    from sklearn.linear_model import SGDClassifier
    from sklearn.cross_validation import KFold

    df = calcFeats('ali-1.9.hdf', 'mfc-1.9.hdf', 'expr3.txt', perphone=False)
    kf = KFold(n=len(df), n_folds=10, shuffle=True)
    args = [[df, train, test, attrs] for train, test in kf]

    def svm(args):
        df, train, test, attrs = args
        dftrain = df.iloc[train]
        dftest = df.iloc[test]
        clf = SGDClassifier(loss='squared_hinge', penalty='l2')
        clf.fit(dftrain.loc[:, attrs].values, dftrain.lbl.values)
        clf.score(dftest.loc[:, attrs].values, dftest.lbl.values)

    pool = mp.Pool(mp.cpu_count())
    pool.map(svm, args)
    pool.close()
    pool.terminate()
    pool.join()

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

