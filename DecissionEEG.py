
# ==============================================================================
# Copyright (c) 2019, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
# Author: , Waldo Nogueira (NogueiraVazquez.Waldo@mh-hannover.de), Hanna Dolhopiatenko (Dolhopiatenko.Hanna@mh-hannover.de)
# All rights reserved.
# ==============================================================================
'''This code represents decission architecture. Only used the lagged EEG as input without speech envelopes'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import scipy.io as io
import random
from LoadData2 import loadData2
from tensorflow.keras import optimizers
from math import floor
from math import ceil




'''Create model'''

def createModel():
    
    model = Sequential()
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dense(n_hidden**(5), input_shape=(Window10s, (numChans+2)*2), activation='relu', use_bias=True))    
    model.add(Dropout(dropout))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dense(n_hidden**(4),  activation='relu', use_bias=True))    
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='tanh'))  

    model.compile(loss='mean_squared_error', optimizer='adam')
   
    return model

'''Define necessary parameters'''
fs=64
n_hidden=2 
dropout=0.25
Window10s=640
numBlocks=288  #EEG Signal has 288 blocks, 10s each 

'''############## MAIN CODE ################'''

for Subject in range(1,2):
  
    workingDir='E:/HomeOffice/CodeforGitHub'  #Provide your own working path here


    '''Load Dataset'''
    a, b, chans, transfer1=loadData2(Subject)
    numChans=len(chans)
    numSides=2
    transfer2 =io.loadmat(workingDir+'/wav.mat');
    LAGS15 = [250]

    eegData=np.zeros((numBlocks,Window10s,numChans+2),dtype=np.float32)
    targetAudio=np.zeros((numBlocks,Window10s,numSides),dtype=np.float32)
    eegDataTMP=np.zeros((Window10s,numChans),dtype=np.float32)
    eegDataTMP2=np.zeros((Window10s,numChans+1),dtype=np.float32)
    envTMPA = np.zeros((Window10s,1),dtype=np.float32) 
    envTMPU = np.zeros((Window10s,1),dtype=np.float32) 
    
    '''Split Dataset in 288 blocks'''
    for block in range(numBlocks):
        eegDataTMP=transfer1['eeg'][block*Window10s:(block+1)*Window10s,:]

        blockES = int(np.floor(block/6))
        if a[blockES]==1:
             envTMPA = transfer2["EnvA"][block*Window10s:(block+1)*Window10s,0]  
             envTMPU = transfer2["EnvU"][block*Window10s:(block+1)*Window10s,0]
        else:
             envTMPA = transfer2["EnvU"][block*Window10s:(block+1)*Window10s,0]
             envTMPU = transfer2["EnvA"][block*Window10s:(block+1)*Window10s,0]

        eegDataTMP2 = np.concatenate((eegDataTMP,envTMPA[:,None]),axis=1)     
        eegData[block,:,:] = np.concatenate((eegDataTMP2,envTMPU[:,None]),axis=1)
        
        targetAudio[block,:,0]=np.ones((Window10s),dtype=np.float32)
        targetAudio[block,:,1]=np.zeros((Window10s ),dtype=np.float32)
    
   
    ''' Choose random blocks for Training/Validation/Testing'''
    leaveBlocks= random.sample(range(287), 144)
    leaveValidat=leaveBlocks[:72] 
    leaveTest=leaveBlocks[72:]
    
    '''Training Dataset'''
    trainingDataEEG=np.zeros(((numBlocks-2),Window10s,numChans+2)) 
    trainingDataAudioA=np.zeros(((numBlocks-2),Window10s,1))
    trainingDataAudioU=np.zeros(((numBlocks-2),Window10s,1))
    trainingDataEEGlagged=np.zeros(((numBlocks-2),Window10s,numChans+2))
    
    '''Validation Set'''
    develDataEEG=np.zeros(((numBlocks-287),Window10s,numChans+2))
    develDataAudioA=np.zeros(((numBlocks-287),Window10s,1))
    develDataAudioU=np.zeros(((numBlocks-287),Window10s,1))
    

    '''Testing Set'''   
    testDataEEG=np.zeros(((numBlocks-287),Window10s,numChans+2))
    testDataAudioA=np.zeros(((numBlocks-287),Window10s,1))
    testDataAudioU=np.zeros(((numBlocks-287),Window10s,1))

    
    StartLagTrain=np.zeros(((numBlocks-2),Window10s,numChans+2))
    EndLagTrain=np.zeros(((numBlocks-2),Window10s,numChans+2))
    StartLagDevel=np.zeros(((numBlocks-287),Window10s,numChans+2))
    EndLagDevel=np.zeros(((numBlocks-287),Window10s,numChans+2))
    StartLagTest=np.zeros(((numBlocks-287),Window10s,numChans+2))
    EndLagTest=np.zeros(((numBlocks-287),Window10s,numChans+2)) 
    results1=np.zeros((5))*np.nan

    predicted = np.zeros((len(LAGS15),numBlocks, 2))
    tested = np.zeros((len(LAGS15),numBlocks, 2))


    lags_length=len(LAGS15)
    for end_lagi in range(len(LAGS15)):
        print(end_lagi)
        end_lag=LAGS15[end_lagi]
        start_lag=end_lag-15    
        start=start_lag
        fin=end_lag
        start=floor(start/1e3*fs)
        fin=ceil(fin/1e3*fs)

        for blockCV in range(len(leaveValidat)):
            leaveValidat11=leaveValidat[blockCV]
            leaveTest11=leaveTest[blockCV]
            i=0
            for block in range(numBlocks):
                if leaveValidat11==block or leaveTest11==block:
                    continue
                trainingDataEEG[i,:,:]=eegData[block,:,:]
                blockE = int(np.floor(block/6))
                trainingDataAudioA[i,:,0]=targetAudio[block,:,b[blockE]]
                trainingDataAudioU[i,:,0]=targetAudio[block,:,a[blockE]]
                i+=1
            k=0
            develDataEEG[:,:,:]=eegData[leaveValidat11,:,:]
            blockV = int(np.floor(leaveValidat11/6))
            develDataAudioA[:,:,0]=targetAudio[leaveValidat11,:,b[blockV]]
            develDataAudioU[:,:,0]=targetAudio[leaveValidat11,:,a[blockV]] 
            testDataEEG[:,:,:]=eegData[leaveTest11,:,:]
            blockT = int(np.floor(leaveTest11/6))
            testDataAudioA[:,:,0]=targetAudio[leaveTest11,:,b[blockT]]
            testDataAudioU[:,:,0]=targetAudio[leaveTest11,:,a[blockT]]
            
            '''Lag EEG Signal'''
            StartLagDevel[k,:,:]= np.pad(develDataEEG[k,:,:], ((0, start), (0, 0)), mode='constant')[start:, :]
            EndLagDevel[k,:,:]=np.pad(develDataEEG[k,:,:], ((0, fin), (0, 0)), mode='constant')[fin:, :]
            DevelDataEEGLagged=np.concatenate([StartLagDevel, EndLagDevel], axis=2)
            StartLagTest[k,:,:]= np.pad(testDataEEG[k,:,:], ((0, start), (0, 0)), mode='constant')[start:, :]
            EndLagTest[k,:,:]=np.pad(testDataEEG[k,:,:], ((0, fin), (0, 0)), mode='constant')[fin:, :]
            TestDataEEGLagged=np.concatenate([StartLagTest, EndLagTest], axis=2)
    
            for block in range(numBlocks-2):
                StartLagTrain[block,:,:] = np.pad(trainingDataEEG[block,:,:], ((0, start), (0, 0)), mode='constant')[start:, :]
                EndLagTrain[block,:,:]   = np.pad(trainingDataEEG[block,:,:], ((0, fin), (0, 0)), mode='constant')[fin:, :]
                TrainingDataEEGLagged=np.concatenate([StartLagTrain, EndLagTrain], axis=2)
       
            '''Create and fit model'''
            Model=createModel()
            tempModelName=workingDir+'/RevSingle.hdf5'
            checkLow = ModelCheckpoint(filepath=tempModelName, verbose=0, save_best_only=True,mode='min',monitor='val_loss')            
            early = EarlyStopping(monitor='val_loss',patience=10, mode='min')
            
            trainingDataAudio = np.concatenate((trainingDataAudioA[:,:,:],trainingDataAudioU[:,:,:]),axis=2)
            develDataAudio = np.concatenate((develDataAudioA[:,:,:],develDataAudioU[:,:,:]),axis=2)
            testDataAudio = np.concatenate((testDataAudioA[:,:,:],testDataAudioU[:,:,:]),axis=2)
         
            Model.fit(TrainingDataEEGLagged[:,:,:],trainingDataAudio[:,:,:],batch_size=2,epochs=150,verbose=1,callbacks=[early,checkLow],validation_data=(DevelDataEEGLagged[:,:,:],develDataAudio[:,:,:]))
            Model.load_weights(tempModelName)

            '''Prediction'''
            predictionA=Model.predict(TestDataEEGLagged[:,:,:])    
            predicted[end_lagi,blockCV,:] = np.mean(predictionA,axis=1)
            tested[end_lagi,blockCV,:] = np.mean(testDataAudio,axis=1)


io.savemat(workingDir+'/Results/RevMulti_'+str(Subject)+'.mat',{'predicted'+str(Subject):predicted, 'Growntrouth'+str(Subject):tested)    
           
            
