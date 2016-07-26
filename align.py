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

def makeMfcHdf(mfc_dir, hdf_file):
    with h5py.File(hdf_file, 'w') as f:
        for s in os.listdir(mfc_dir):
            g = f.create_group(s[0]+s[6:9])
            spk = np.load(os.path.join(mfc_dir, s))
            for n, d in spk.iteritems():
                g.create_dataset(name=n, data=d[::-1])
    
# this can't actually be automated cos it depends on the semantics of the data
def makeAliHdf(ali_file, hdf_file):
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

def calcFormantsParallel(args):
    spkpath, outhdf, alihdf = args
    spkr = spkpath.split('/')[-1]
    # get frame size
    sr = 16000        # samplerate
    fss = 0.01        # framesize in seconds
    hpf = 50.         # highpass freq in Hz
    fs = int(fss*sr)  # framesize in samples
    # calc hanning window
    win = np.hanning(fs)
    # calc highpass filter coefficients
    b, a = butter(1, 50./(0.5*sr), "highpass")

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

    def alignPhones(data, stride):
        align = []
        for i, s in enumerate(stride):
            e = stride[i+1] if (i+1) < stride.size else data.shape[-1]
            align.append(np.mean(data[:,s:e], axis=-1))
        return np.array(align)

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

    # get wav files in specified directory
    wavsin = []
    argout = []
    for dirpath,_,filename in os.walk(spkpath):
        wavsin.extend([os.path.join(dirpath,f) for f in filename if f.split('.')[-1] == 'wav'])
        argout.extend([f.replace('.wav','') for f in filename if f.split('.')[-1] == 'wav'])
    paths = dict(zip(argout, wavsin))
    
    ali = h5py.File(alihdf)
    out = h5py.File(outhdf)
    try:
        alispk = ali[spkr]
    except:
        print(spkr + ' not in alignment file')
        return
    try:
        outspk = out.create_group(spkr)
    except:
        outspk = out[spkr]
    files = list(set(alispk.keys()) & set(argout))
    forfeat = []
    alifeat = []
    allf = []
    for f in files:
        path = paths[f]
        if not os.path.exists(path):
            print(f + '.wav does not exist')
            continue
        if f in outspk.keys():
            wavfeat = outspk[f]
        else:
            # calc wave feats: f0-f4 + energy: this is VERY SLOW
            wavfeat = calcWaveFeats(path, fs, win, b, a)
            outspk.create_dataset(name=f, data=wavfeat)
        # align formant and timing data: DON'T CHANGE CONSTANTS
        alidata = alispk[f].value
        strides = (200*alidata[:,1]).astype(int)
        forfeat.extend(alignPhones(wavfeat, strides))
        alifeat.extend(alidata[:,[0,2]])
        allf.extend([f]*len(strides))

    # recreate delta and delta2 features (like in MFCC)
    alifeat = np.array(alifeat)
    forfeat = np.array(forfeat)
    delfor = np.diff(forfeat, axis=0)
    deldelfor = np.diff(delfor, axis=0)
    zero = np.zeros((1, forfeat.shape[-1]), dtype=float)
    delfor = np.vstack((delfor, zero))
    deldelfor = np.vstack((deldelfor, zero, zero))
    forfeat = np.hstack((forfeat, delfor, deldelfor))

    try:
        feats = pd.DataFrame(np.hstack((alifeat, forfeat)), index=allf, columns=['phon', 'dur']+range(forfeat.shape[-1]))
        sys.stdout.flush()
        print(spkr, end=" ")
        return feats
    except:
        print(spkr, forfeat.shape, alifeat.shape, len(allf))
        return
        #return forfeat, alifeat, allf

def calcFormants(wavpath, outfile, alifile):
    #return calcFormantsParallel([wavpath, outfile, alifile])
    args = [[os.path.join(wavpath, path), outfile, alifile] for path in os.listdir(wavpath)]
    pool = mp.Pool(mp.cpu_count())
    formants = pool.map(calcFormantsParallel, args)
    pool.close()
    pool.terminate()
    pool.join()

    return formants

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

    # recreate MFCC features
    mfcfeat = np.array(mfcfeat)
    delmfc = np.diff(mfcfeat, axis=0)
    deldelmfc = np.diff(delmfc, axis=0)
    zero = np.zeros((1, 13), dtype=float)
    delmfc = np.vstack((delmfc, zero))
    deldelmfc = np.vstack((deldelmfc, zero, zero))
    mfcfeat = np.hstack((mfcfeat, delmfc, deldelmfc))

    """
    numfeats = mfcfeat.shape[1]
    # delta wave features: average distance of a phone from the succeeding
    #ephone
    delfeats = np.sum(np.square(np.diff(mfcfeat[:, :-1],axis=0)),axis=-1)**0.5
    delfeats = np.array(delfeats.tolist() + [0.0]).reshape((delfeats.shape[0]+1, 1))
    """
    # horizontally stack all the data together
    feats = np.concatenate((alifeat, mfcfeat), axis=1)
    # make dataframe from features
    dfspk = pd.DataFrame(feats, index=allf, columns=['phon','dur']+range(39))

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

def alignParallel(chunk):
    ali = h5py.File('ali-2.0.hdf')
    mfc = h5py.File('mfc-2.0.hdf')
    aligned = []
    for f in chunk:
        a = ali[f].value
        if a.ndim == 1: continue
        m = mfc[f].value
        e = a[:,0]+a[:,1]
        ma = [m[a[n,0]:e[n]] for n in xrange(a.shape[0]) if m.size > 0]
        ma = np.array([np.hstack((md.mean(axis=0), np.diff(md, axis=0).mean(axis=0), np.diff(md, n=2, axis=0).mean(axis=0))) for md in ma])
        aligned.append(pd.DataFrame(np.hstack((a, ma)), index=[f]*a.shape[0]))
    ali.close()
    mfc.close()
    return pd.concat(aligned)

def alignMFCC(alifile='ali-2.0.hdf', mfcfile='mfc-2.0.hdf', chunksize=1000):
    ali = h5py.File(alifile)
    mfc = h5py.File(mfcfile)
    files = ali.keys()
    chunks = [files[i:i+chunksize] for i in xrange(0, len(files), chunksize)]
    ali.close()
    mfc.close()

    pool = mp.Pool(mp.cpu_count())
    aligned = pool.map(alignParallel, [c for c in chunks])
    pool.close()
    pool.terminate()
    pool.join()
    return pd.concat(aligned)

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

    # join and normalize
    df = pd.concat(feats)
    cols = dict(zip(range(len(df.columns)), ['spkr', 'k', 's', 'u', 'snum', 'rnum', 'sfreq', 'rfreq', 'sdur'] + ['s'+str(n) for n in range(39)] + ['rdur'] + ['r'+str(n) for n in range(39)]))
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

def classify(data, lbl, clf='svm', sample=False, ret=False, plot=False, scoring='f1'):
    from sklearn.cross_validation import KFold
    from sklearn.cross_validation import cross_val_score
    #from sklearn.metrics import f1_score

    if clf == 'svm':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='hinge', penalty='l1', n_iter=100)
    elif clf == 'log':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='log', penalty='l1', n_iter=100)
    # ensemble methods
    elif clf == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=25, max_depth=None, min_samples_split=1, n_jobs=4)
    elif clf == 'gb':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=25, learning_rate=1.0)

    df = data.dropna()
    if sample:
        dfs = df.loc[lbl[lbl == True].index].sample(1000)
        dfr = df.loc[lbl[lbl == False].index].sample(1000)
        df = pd.concat((dfs, dfr))
        lbl = lbl[df.index]
        if plot:
            import plots
            plots.scatter2d(df)

    kf = KFold(len(df), n_folds=10, shuffle=True, random_state=1)
    if ret:
        scores = cross_val_score(clf, df, lbl, scoring=scoring, cv=kf, n_jobs=4)
        print(scores)
        return scores

    misses = []
    wrongs = []
    scores = []
    for r, s in kf:
        dfr = df.iloc[r]
        dfs = df.iloc[s]
        clf.fit(dfr.values, lbl[r])
        predict = clf.predict(dfs.values)
        score = predict != lbl[s]
        scores.append(100. - 100.*len(np.argwhere(score))/len(dfs))
        wrongs.append(dfs.iloc[(np.argwhere(score.values)).flatten()])
        misses.append(score)

    misses = pd.concat(misses)
    wrongs = pd.concat(wrongs)
    return misses, wrongs, scores, clf

def extractParallel(chunk):
    s = chunk.groupby(chunk.index)
    ds = s.transform(lambda x:x.diff()).groupby(chunk.index)
    dds = ds.transform(lambda x:x.diff()).groupby(chunk.index)

    smean = s.mean()
    sdev = s.std()
    dsmean = ds.mean()
    dsdev = ds.std()
    ddsmean = dds.mean()
    ddsdev = dds.std()
    
    feat = pd.concat((smean, sdev, dsmean, dsdev, ddsmean, ddsdev), axis=1)
    return feat

def extractFeats(df, chunksize=1000):
    files = df.index.unique()
    chunks = [files[i:i+chunksize] for i in xrange(0, len(files), chunksize)]

    pool = mp.Pool(mp.cpu_count())
    feats = pool.map(extractParallel, [df.loc[c] for c in chunks])
    pool.close()
    pool.terminate()
    pool.join()
    return pd.concat(feats)

def getAcFeats(df):
    def normalize(x):
        return (x - x.mean())/x.std()

    dfg = df.groupby(df.index).transform(normalize).groupby(df.index)

    # calculate speed of pronunciatioin
    #spd = dfg.dur.count().astype(float)/dfg.dur.sum()

    # take normalized mean and variance of segments per utterance
    dfm = dfg.mean()
    #dfv = dfg.std()

    # take normalized mean and variance of differences between consecutive
    # segments per utterance or "delta segments"
    ds = dfg.transform(lambda x:x.diff()).groupby(df.index)
    dsm = ds.mean()
    dsv = ds.std()

    # take delta delta segments
    dds = ds.transform(lambda x:x.diff()).groupby(df.index)
    ddsm = dds.mean()
    ddsv = dds.std()

    fe = pd.concat((dfm, dsm, dsv, ddsm, ddsv), axis=1)
    fe.dropna(inplace=True)

    return fe

def trainAll(fe):
    scores = []
    f1s = []
    for c in ['svm', 'log', 'rf', 'gb']:
        f1 = classify(fe.iloc[:,:-1], fe.lbl, clf=c, sample=True, scoring='f1', ret=True)
        score = classify(fe.iloc[:,:-1], fe.lbl, clf=c, sample=True, scoring='accuracy', ret=True)
        f1s.append([np.mean(f1), np.std(f1)*2])
        scores.append([np.mean(score), np.std(score)])
    return scores, f1s

def predictSponRead(attrs):
    from sklearn.linear_model import SGDClassifier
    from sklearn.cross_validation import KFold

    df = calcFeats('ali-1.9.hdf', 'mfc-1.9.hdf', 'expr3.txt', perphone=False)
    kf = KFold(n=len(df), n_folds=10, shuffle=True)
    args = [[df, train, test, attrs] for train, test in kf]

    pool = mp.Pool(mp.cpu_count())
    pool.map(svm, args)
    pool.close()
    pool.terminate()
    pool.join()

def mapPhones(df):
    pm = {}
    for p in open('phones.txt').read().split('\n')[:-1]:
        i, n = p.split()
        pm[int(n)] = i
    df.phon = df.phon.map(pm)

def groupDiff(dfin, by):
    from scipy.spatial.distance import cdist
    df = dfin.copy()
    # get used phones
    mapPhones(df)
    idx = pd.read_pickle('allpp.pk').index
    df = df.loc[df.phon.isin(idx)]
    v = df.iloc[:,2:-1]
    df.iloc[:,2:-1] = (v-v.mean())/v.std()

    # get center
    c = df.groupby([df.phon]).mean().drop(['lbl'], axis=1)
    print(c.head())
    # group by
    pp = df.groupby([df.spkr.str.get(by), df.phon]).mean()
    print(pp.head())

    res = []
    groups = pp.index.get_level_values(0).unique()
    for n in groups:
        if agg:
            d = pp.xs(n, level=0)
            # per phone distances to center
            #d -= c
            # get per group euclidean distance between spon/read phone pairs
            d = d.xs(True, level=0) - d.xs(False, level=0)
            d = (d**2).sum(axis=1)**0.5
            #d = d.xs(True, level=0)/d.xs(False, level=0)
        else:
            d = pp.xs(n, level=0)
            d = d.xs(True, level=0) - d.xs(False, level=0)
            d.columns = pd.MultiIndex.from_tuples(zip([n]*len(d.columns), d.columns))
        res.append(d)

    if agg:
        res = pd.DataFrame(res, index=groups).T
    else:
        res = pd.concat(res, axis=1)
    return res
def spectralReduction(dfin, by, agg=True):
    from scipy.spatial.distance import cdist
    df = dfin.copy()
    # get used phones
    mapPhones(df)
    idx = pd.read_pickle('allpp.pk').index
    df = df.loc[df.phon.isin(idx)]
    if agg:
        v = df.iloc[:,2:-1]
        df.iloc[:,2:-1] = (v-v.mean())/v.std()

    # get center
    c = df.groupby([df.phon]).mean().drop(['lbl'], axis=1)
    print(c.head())
    # group by
    pp = df.groupby([df.spkr.str.get(by), df.lbl, df.phon]).mean()
    print(pp.head())

    res = []
    groups = pp.index.get_level_values(0).unique()
    for n in groups:
        if agg:
            d = pp.xs(n, level=0)
            # per phone distances to center
            #d -= c
            # get per group euclidean distance between spon/read phone pairs
            d = d.xs(True, level=0) - d.xs(False, level=0)
            d = (d**2).sum(axis=1)**0.5
            #d = d.xs(True, level=0)/d.xs(False, level=0)
        else:
            d = pp.xs(n, level=0)
            d = d.xs(True, level=0) - d.xs(False, level=0)
            d.columns = pd.MultiIndex.from_tuples(zip([n]*len(d.columns), d.columns))
        res.append(d)

    if agg:
        res = pd.DataFrame(res, index=groups).T
    else:
        res = pd.concat(res, axis=1)
    return res

def distanceGraph(df):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pm = {}
    for p in open('phones.txt').read().split('\n')[:-1]:
        i, n = p.split()
        pm[int(n)] = i

    # group by phone and label and then normalize
    dfp = df.groupby([df.phon.astype(int).map(pm), df.lbl]).mean().drop(['phon'], axis=1).apply(lambda x:x/x.max())

    # reduce to 2 dimension for profitable plotting
    pca = PCA(n_components=2)
    dfl = pd.DataFrame(pca.fit_transform(dfp), index=dfp.index)
    # remove phonemes missing a label (only has spontaneous or only has read
    # data)
    dfl = dfl.loc[list(set(dfl.xs(True, level=1).index) & set(dfl.xs(False, level=1).index))]
    # calculate distances
    dist = dfl.groupby(level=0).transform(lambda x:x.diff()).dropna()
    dist['d'] = np.sqrt(np.sum((dfl.values)**2, axis=1))
    s = pca.transform(dfs.values)
    r = pca.transform(dfr.values)
    x = np.hstack((s[:,0], r[:,0]))
    y = np.hstack((s[:,1], r[:,1]))
    [plt.plot(x[n], y[n]) for n in range(x.shape[0])]
    return x, y

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

