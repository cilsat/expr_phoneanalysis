import numpy as np
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
