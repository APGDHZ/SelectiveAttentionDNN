
"""
This file is loading EEG Data
"""

import scipy.io as io


workingDir=''   #Provide path to your datasets

# a and b defines to which side subject was attending. For more details read READM

def loadData2(Subject):
    if Subject==1:
        a=[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]  
        b=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
        chans=list(range(96))  
        transfer1 = io.loadmat(workingDir+'/eeg_example.mat')

    return a, b, chans, transfer1