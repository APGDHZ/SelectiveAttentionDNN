

# ==============================================================================
# Copyright (c) 2019, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
# Author: Hanna Dolhopiatenko (Dolhopiatenko.Hanna@mh-hannover.de), Waldo Nogueira (NogueiraVazquez.Waldo@mh-hannover.de)
# All rights reserved.
# ==============================================================================
'''This code represents fully connected neural network. EEG Data divided in 48 blocks. 46 blocks are taken for training, one for validation and another one for testing.
10 random block for test and validation were chosed to provide cross-validation alghoritm'''



import tensorflow 
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import scipy.io as io
import random
from LoadData import loadData  #This module you have to modify to load your own data. The example is provided at the same repository
from tensorflow.python.keras import optimizers
from math import floor
from math import ceil


'''Define loss function, based on correlation'''   
def corr_loss(act,pred):   
    cov=(K.mean((act-K.mean(act))*(pred-K.mean(pred))))
    return 1-(cov/(K.std(act)*K.std(pred)+K.epsilon()))


'''Create model'''

def createModel():
    model = Sequential()
    model.add(Dense(n_hidden**(3), input_shape=(trainPredWin,numChans*2), activation='relu', use_bias=True))    #Input layer 
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(dropout))
    model.add(Dense(n_hidden**(2),  activation='relu', use_bias=True))    
    model.add(Dropout(dropout))
    model.add(Dense(n_hidden**(1), activation='relu', use_bias=True))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='tanh')) 
    model.compile(loss=corr_loss, optimizer='adam')
    return model

'''Define the necessary parameters'''

fs=64   #sampling rate of processed EEG and Audio signals
n_hidden=2 
dropout=0.25
trainPredWin=60*fs    #Training prediction Window
numBlocks=48          #EEG Signal divided in 48 blocks, 60s each



'''############## MAIN CODE ################'''
for Subject in range(1,2):       #Choose which subject to process
    workingDir='E:\dolhopia\DNN\EasyModel'  #Provide your own working path here
    a, b, chans, eeg=loadData(Subject)   #Load Dataset. a and b show to which stream subject was attending
    numChans=len(chans)     
    numSides=2
    Audio =io.loadmat(workingDir+'/Data/wav.mat');
      
    LAGS15 = [250]        #Define the Lag
 
    
    eegData=np.zeros((numBlocks,trainPredWin,numChans),dtype=np.float32)
    targetAudio=np.zeros((numBlocks,trainPredWin,numSides),dtype=np.float32)
    
    '''Split Dataset in 48 blocks'''
    for block in range(numBlocks):
        eegData[block,:,:]=eeg['eegNH'+str(Subject)][block*trainPredWin:(block+1)*trainPredWin,:]
        targetAudio[block,:,a]=Audio["EnvA"][block*trainPredWin:(block+1)*trainPredWin,0] #Here you need to Load the envelopes for the attended signal
        targetAudio[block,:,b]=Audio["EnvU"][block*trainPredWin:(block+1)*trainPredWin,0] #Here you need to Load the envelopes for the unattended signal
    
    ''' Choose random blocks for Training/Validation/Testing'''
    leaveBlocks= random.sample(range(47), 10)
    leaveValidat=leaveBlocks[:5] 
    leaveTest=leaveBlocks[5:]

    '''Training Dataset'''
    trainingDataEEG=np.zeros(((numBlocks-2),trainPredWin,numChans)) # 80% of data for training
    trainingDataAudioA=np.zeros(((numBlocks-2),trainPredWin,1))
    trainingDataAudioU=np.zeros(((numBlocks-2),trainPredWin,1))
    trainingDataEEGlagged=np.zeros(((numBlocks-2),trainPredWin,numChans))

    '''Validation Set'''
  
    develDataEEG=np.zeros(((numBlocks-47),trainPredWin,numChans))
    develDataAudioA=np.zeros(((numBlocks-47),trainPredWin,1))
    develDataAudioU=np.zeros(((numBlocks-47),trainPredWin,1))

    '''Testing Set'''
    testDataEEG=np.zeros(((numBlocks-47),trainPredWin,numChans))
    testDataAudioA=np.zeros(((numBlocks-47),trainPredWin,1))
    testDataAudioU=np.zeros(((numBlocks-47),trainPredWin,1))
            
    '''Different Sets'''
    StartLagTrain=np.zeros(((numBlocks-2),trainPredWin,numChans))
    EndLagTrain=np.zeros(((numBlocks-2),trainPredWin,numChans))
    StartLagDevel=np.zeros(((numBlocks-47),trainPredWin,numChans))
    EndLagDevel=np.zeros(((numBlocks-47),trainPredWin,numChans))
    StartLagTest=np.zeros(((numBlocks-47),trainPredWin,numChans))
    EndLagTest=np.zeros(((numBlocks-47),trainPredWin,numChans)) 
    results1=np.zeros((len(LAGS15),5))*np.nan
    corrCoefA=np.zeros((len(LAGS15),numBlocks, 1))
    corrCoefU=np.zeros((len(LAGS15),numBlocks, 1))
    Acc2=np.zeros((len(LAGS15), 1))

        
    lags_length=len(LAGS15)
    for end_lagi in range(len(LAGS15)):     #Start loop across Lags
        print(end_lagi)
        end_lag=LAGS15[end_lagi]
        start_lag=end_lag-15    
        start=start_lag
        fin=end_lag
        start=floor(start/1e3*fs)
        fin=ceil(fin/1e3*fs)
        for blockCV in range(46):           #Cross Validation
            leaveValidat11=leaveValidat[blockCV]
            leaveTest11=leaveTest[blockCV]
            i=0
            for block in range(numBlocks):
                if leaveValidat11==block or leaveTest11==block:
                    continue
                trainingDataEEG[i,:,:]=eegData[block,:,:]
                trainingDataAudioA[i,:,0]=targetAudio[block,:,b] 
                i+=1
            '''To lag the EEG Dataset'''
            k=0
            develDataEEG[:,:,:]=eegData[leaveValidat11,:,:]
            develDataAudioA[:,:,0]=targetAudio[leaveValidat11,:,b] 
            testDataEEG[:,:,:]=eegData[leaveTest11,:,:]
            testDataAudioA[:,:,0]=targetAudio[leaveTest11,:,b] 
            testDataAudioU[:,:,0]=targetAudio[leaveTest11,:,a]
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

            '''Create Model'''
            Model=createModel()
            tempModelName=workingDir+'/model/ModelWeights.hdf5'  #Save weights
            checkLow = ModelCheckpoint(filepath=tempModelName, verbose=0, save_best_only=True,mode='min',monitor='val_loss')            
            early = EarlyStopping(monitor='val_loss',patience=10, mode='min')  #Early Stopping to get to the point in which the loss on the Development set does not decrease anymore

            Model.fit(TrainingDataEEGLagged[:,:,:],trainingDataAudioA[:,:,:],batch_size=2,epochs=300,verbose=1,callbacks=[early,checkLow],validation_data=(DevelDataEEGLagged[:,:,:],develDataAudioA[:,:,:]))
            Model.load_weights(tempModelName)

            '''Prediction'''
            predictionA=Model.predict(TestDataEEGLagged[:,:,:])   
            
            '''Correlate with Original Audio'''
            corrCoefA[end_lagi,blockCV,:] = np.corrcoef(testDataAudioA[k,:,0],predictionA[k,:,0])[1,0]
            corrCoefU[end_lagi,blockCV,:] = np.corrcoef(testDataAudioU[k,:,0],predictionA[k,:,0])[1,0]
        '''Calculate Accuracy'''    
        Acc2[end_lagi,:] = np.mean(corrCoefU[end_lagi,:,:]<corrCoefA[end_lagi,:,:])

io.savemat(workingDir+'/Results/RevMulti_'+str(Subject)',{'corrCoeffA'+str(Subject):corrCoefA, 'corrCoefU'+str(Subject):corrCoefU, 'Acc2'+str(Subject):Acc2}) 
           

    
    




