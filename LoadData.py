
"""
This file is loading EEG Data
"""

import scipy.io as io

workingDir='' #Provide here path to your data

def loadData(Subject):
    if Subject==1:
        a=1
        b=0
        chans=list(range(96))  
        transfer1 = io.loadmat(workingDir+'/Data/eeg_example.mat')
    return a, b, chans, transfer1