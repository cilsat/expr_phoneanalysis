import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def demographics(df):
    dfs = df.groupby([df.spkr.str[:-4]]).mean()
    dfsg = dfs.groupby([dfs.index.str.get(4), dfs.index.str.get(3)]).count().dur
    print(dfsg.head(20))

    suku = dfsg.index.get_level_values(0).unique()
    N = np.arange(len(suku))
    w = 0.35

    b1 = plt.bar(N, dfsg.xs('L', level=1), w, color='0.5')
    b2 = plt.bar(N+w, dfsg.xs('P', level=1), w, color='0.25')
    plt.xticks(N+w, suku)
    plt.legend(('men', 'women'), loc=1)
    plt.xlabel('Dialects by gender')
    plt.ylabel('Number of speakers')

    plt.show()

def hitrate(df, s, nbins=25):
    dft = df.loc[s.index]
    dft = dft.groupby(dft.index).count().dur
    hr = pd.concat((dft, s), axis=1)
    hr.columns = ['seg', 'miss']
    print(hr.head())

    hr = hr.loc[hr.seg < 130]
    _, bins = np.histogram(hr.seg, nbins)
    h1, _ = np.histogram(hr.loc[hr.miss == False], bins=bins)
    h2, _ = np.histogram(hr.loc[hr.miss == True], bins=bins)

    N = bins[:-1]
    print(bins)
    w = np.diff(N)[0]
    #b1 = plt.bar(N, h1, w, color='0.85')
    #b2 = plt.bar(N, h2, w, color='0.65')
    l1 = plt.plot(N, (h1.astype(float)/h2)**2, 'k', linewidth=1.5)
    ls = plt.plot(N, h1, 'k-.')
    lr = plt.plot(N, h2, 'k--')
    plt.ylabel('Number of hits/misses')
    plt.xlabel('Number of segments in utterance')
    plt.yscale('log')
    plt.legend(('hitrate', 'hits', 'misses'))
    plt.show()

def segdist(df, nbins=148):
    # count segments per utterance
    dfs = df.groupby([df.index, df.lbl]).dur.count()
    # discard outliers
    dfs = dfs[dfs < nbins+2]
    # make histogram bins
    _, bins = np.histogram(dfs, nbins)
    s = dfs.xs(True, level=1)
    r = dfs.xs(False, level=1)
    hs, _ = np.histogram(s, bins=bins)
    hr, _ = np.histogram(r, bins=bins)
    #hs = hs.astype(float)/len(s)
    #hr = hr.astype(float)/len(r)

    N = bins[:-1]
    w = np.diff(N)[0]*0.5
    print(bins)
    #bs = plt.bar(N, hs, w, color='0.65')
    #br = plt.bar(N+w, hr, w, color='0.35')
    plt.plot(N, hs, 'k--')
    plt.plot(N, hr, 'k.')
    plt.yscale('log')
    plt.ylabel('Number of utterances')
    plt.xlabel('Number of segments in utterance')
    plt.legend(('spontan', 'read'))
    plt.show()

def scatter2d(df, n=100):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    df.dropna(inplace=True)
    pca.fit(df.loc[:, set(df.columns) - set('lbl')])

    s = df.loc[df.lbl == True].sample(n)
    r = df.loc[df.lbl == False].sample(n)

    ss = pca.transform(s.loc[:, set(df.columns) - set('lbl')])
    rr = pca.transform(r.loc[:, set(df.columns) - set('lbl')])

    s1 = plt.scatter(ss[:,0], ss[:,1], color='0.5', marker='D')
    s2 = plt.scatter(rr[:,0], rr[:,1], color='k', marker='x')
    plt.legend((s1, s2), ('spontan', 'read'))
    plt.show()

def scatter3d(df, n=100):
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=3)
    df.dropna(inplace=True)
    pca.fit(df.loc[:, set(df.columns) - set('lbl')])

    s = df.loc[df.lbl == True].sample(n)
    r = df.loc[df.lbl == False].sample(n)

    ss = pca.transform(s.loc[:, set(df.columns) - set('lbl')])
    rr = pca.transform(r.loc[:, set(df.columns) - set('lbl')])

    ax = plt.gca(projection='3d')
    s1 = ax.scatter(ss[:,0], ss[:,1], ss[:,2], color='r')
    s2 = ax.scatter(rr[:,0], rr[:,1], rr[:,2], color='b')
    plt.legend((s1, s2), ('spontan', 'read'))
    plt.show()

