#!/usr/bin/python
import sys, os
import numpy as np
import pandas as pd

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
    return spon, read

def calcFeats(df):
    #get vowels
    vowels = []
    for p in df['phon'].unique():
        v = p.split('_')[0]
        if v.startswith('a') or v.startswith('e') or v.startswith('i') or v.startswith('o') or v.startswith('u') or v.startswith('@'):
            vowels.append(p)

    # calculate the average and variance of duration of each vowel
    feats = []
    dfsize = 1./df.size
    for v in vowels:
        dur = df[df['phon'] == v].iloc[:,2]
        feats.append([dur.size*dfsize*100, dur.mean(), dur.var()])
    return pd.DataFrame(feats, index=vowels)

def calcSpontanRead(align_file, phones_file):
    spon, read = swapPhones(align_file, phones_file)
    sfeats = calcFeats(spon)
    rfeats = calcFeats(read)
    sfeats.rename(columns={0:'sfreq', 1:'smean', 2:'svar'}, inplace=True)
    rfeats.rename(columns={0:'rfreq', 1:'rmean', 2:'rvar'}, inplace=True)
    feats = pd.concat((sfeats, rfeats), axis=1)
    return feats

if __name__ == "__main__":
    feats = calcSpontanRead(sys.argv[1], sys.argv[2])
    print(feats)

