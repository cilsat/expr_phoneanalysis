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
    for i, a in enumerate(ali):
        s = a.split()
        p = phones[s[-1]]
        ali[i] = [s[0], float(s[2]), float(s[3]), p]

    # make pandas dataframe of filename, location, duration, and name of phone
    ali = pd.DataFrame(ali)
    ali.rename(columns={0:'fil', 1:'loc', 2:'dur', 3:'phon'}, inplace=True)
    return ali

def calcFeats(df, vowels):
    # calculate the average and variance of duration of each vowel
    feats = []
    for v in vowels:
        dur = df[df['phon'] == v].iloc[:,2]
        feats.append([dur.mean(), dur.var()])

if __name__ == "__main__":
    data = swapPhones(sys.argv[1], sys.argv[2])
