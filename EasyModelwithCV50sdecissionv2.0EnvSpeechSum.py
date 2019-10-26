# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:51:40 2019

@author: dolhopia
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:29:15 2019

@author: dolhopia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:12:53 2019

@author: dolhopia
"""

import tensorflow as tf
import keras
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten
#K.tensorflow_backend._get_available_gpus()
import scipy.io as io
import random
from LoadData2 import loadData2
from keras import optimizers
#from sklearn.model_selection import train_test_split





from math import floor
from math import ceil

#import matplotlib.pyplot as plt

import tensorflow as tf
def correlation_coefficient(y_true, y_pred):
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'pearson_r'  in i.name.split('/')]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
        return 1-pearson_r**2

    
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1-K.square(r)


def correlation_coefficient_loss_np(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.sum(np.multiply(xm,ym))
    r_den = np.sqrt(np.multiply(np.sum(np.square(xm)), np.sum(np.square(ym))))
    r = r_num / r_den

    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return 1-np.square(r)
  
    
#

'''Create model'''

def createModel():
    
    model = Sequential()
    #model.add(Flatten())
    model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dense(n_hidden**(5), input_shape=(Window10s, (numChans+1)*2), activation='relu', use_bias=True))    #Input layer (takes 470 ms of EEG)
    
    
    model.add(Dropout(dropout))
    
    model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dense(n_hidden**(4),  activation='relu', use_bias=True))    #Input layer (takes 60 s of EEG)
    model.add(Dropout(dropout))
    
    #model.add(keras.layers.GRU(n_hidden**(2), activation='tanh', recurrent_activation='tanh', use_bias=False, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False))
    
    #model.add(Dense(n_hidden**(2),  activation='relu', use_bias=True))    #Input layer (takes 60 s of EEG)
   # model.add(Dropout(dropout))

   # model.add(Dense(n_hidden**(2),  activation='relu', use_bias=True))    #Input layer (takes 60 s of EEG)
   # model.add(Dropout(dropout))
    
    
    #model.add(Dense(n_hidden**(1), activation='relu', use_bias=True))
    #model.add(Dropout(dropout))
    
    
    #model.add(Dense(trainPredWin, activation='relu'))  #Ouput layer with 1 neuron
    model.add(Dense(2, activation='tanh'))  #Ouput layer with 1 neuron
    #model.add(Dense(1, activation='linear'))  #Ouput layer with 1 neuron
    
    
    #opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(loss = 'logcosh', optimizer = 'nadam')
    #model.compile(loss=correlation_coefficient_loss, optimizer = opt)
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, nesterov=True)
    
    #model.compile(loss=  , optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam')
   
    return model

#Define Cost function for numpy
def np_logcosh(y_true,y_pred):
    return np.mean(np.log(np.cosh(y_true-y_pred)))
#        model.compile('Nadam','logcosh')
 

        

for Subject in range(18,19):
    #range(22,23):        
    '''############## MAIN CODE ################'''
    
    workingDir='C:/Users/nogueiwa/Dropbox/Projects/Selective Attention/NeuralNetwork2.0/Data/8 Hz'
    #O:/DNN/EasyModel
    '''Perameteres'''
    
    fs=64
    n_hidden=2 
    dropout=0.25
    trainPredWin=183040 #Training prediction Window
    Window10s=640
    AllDataSamples=184320
    numBlocks=288  #EEG Signal has 48 blocks, 60s each #184320/60
    
    
    
    '''Load Dataset'''
    
    #Subject=1   #NH1=1, NH2=2.....
    a, b, chans, transfer1=loadData2(Subject)
    
    numChans=len(chans)
    numSides=2
    transfer2 =io.loadmat(workingDir+'/Data/wav.mat');
    
    #LAGS15 = [150, 200, 250, 300]

    LAGS15 = [300]
    
    #eegData=np.zeros((AllDataSamples,numChans),dtype=np.float32)
    #targetAudio=np.zeros((AllDataSamples,numSides),dtype=np.float32)
    
    eegData=np.zeros((numBlocks,Window10s,numChans+1),dtype=np.float32)
    targetAudio=np.zeros((numBlocks,Window10s,numSides),dtype=np.float32)
    
    eegDataTMP=np.zeros((Window10s,numChans),dtype=np.float32)
    eegDataTMP2=np.zeros((Window10s,numChans+1),dtype=np.float32)
    envTMPA = np.zeros((Window10s,1),dtype=np.float32) 
    envTMPU = np.zeros((Window10s,1),dtype=np.float32) 
    
    '''Split Dataset in 48 blocks'''
    for block in range(numBlocks):
        #eegDataTMP=transfer1['eegNH'+str(Subject)][block*Window10s:(block+1)*Window10s,:]
        eegDataTMP=transfer1['eegCI'+str(Subject)][block*Window10s:(block+1)*Window10s,:]
        blockES = int(np.floor(block/6))
        if a[blockES]==1:
             envTMPA = transfer2["EnvA"][block*Window10s:(block+1)*Window10s,0]
             envTMPU = transfer2["EnvU"][block*Window10s:(block+1)*Window10s,0]
        else:
             envTMPA = transfer2["EnvU"][block*Window10s:(block+1)*Window10s,0]
             envTMPU = transfer2["EnvA"][block*Window10s:(block+1)*Window10s,0]
        
        #targetAudio[block,:,0]=transfer2["EnvA"][block*Window10s:(block+1)*Window10s,0] # Waldo: Here you need to Load the envelopes for the attended signal
        #targetAudio[block,:,1]=transfer2["EnvU"][block*Window10s:(block+1)*Window10s,0] # Waldo: Here you need to Load the envelopes for the unattended signal
        
        eegDataTMP2        = np.concatenate((eegDataTMP,envTMPA[:,None]+envTMPU[:,None]),axis=1)     
        eegData[block,:,:] = eegDataTMP2#np.concatenate((eegDataTMP2,envTMPU[:,None]),axis=1)
        
        targetAudio[block,:,0] = np.ones((Window10s),dtype=np.float32)#transfer2["EnvA"][block*trainPredWin:(block+1)*trainPredWin,0] # Waldo: Here you need to Load the envelopes for the attended signal
        targetAudio[block,:,1] = np.zeros((Window10s ),dtype=np.float32)#transfer2["EnvU"][block*trainPredWin:(block+1)*trainPredWin,0] # Waldo: Here you need to Load the envelopes for the unattended signal
    
   # X_train, X_test, y_train, y_test = train_test_split(eegData, targetAudio, test_size=0.993)
    
    #validation_ratio = 0.00347
    #test_ratio = 0.00347
    #train_ratio = 1-2*test_ratio

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
    #x_train, x_test, y_train, y_test = train_test_split(eegData, targetAudio, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
    #x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 


    
    
    ''' Choose random blocks for Training/Validation/Testing'''
    leaveBlocks= random.sample(range(287), 144)
    leaveValidat=leaveBlocks[:72] 
    leaveTest=leaveBlocks[72:]
    
    
    
    #leaveValidat1=list(range(47))
    #leaveTest1=list(range(1,48))
    #leaveTest1.append(0)
    

        
    
    
    '''Training Dataset'''
    #i=0 
    trainingDataEEG=np.zeros(((numBlocks-2),Window10s,numChans+1)) # 80% of data for training
    trainingDataAudioA=np.zeros(((numBlocks-2),Window10s,1))
    trainingDataAudioU=np.zeros(((numBlocks-2),Window10s,1))
    trainingDataEEGlagged=np.zeros(((numBlocks-2),Window10s,numChans+1))
    
    
    #for block in range(numBlocks):
           # if block in leaveBlocks:      #leaveBlocks[:]==block:
           #     continue
           # trainingDataEEG[i,:,:]=eegData[block,:,:]
           # trainingDataAudioA[i,:,0]=targetAudio[block,:,b] 
           # trainingDataAudioU[i,:,0]=targetAudio[block,:,a] 
              
            #i+=1
    
    # Hann: Here I'm trying to extend one frame of the matrix
#    trainingDataEEG2=np.zeros(((numBlocks-10),trainPredWin,2*numChans)) # 80% of data for training
#    for block in range(numBlocks):
#        if block in leaveBlocks:      #leaveBlocks[:]==block:
#                continue
#        for j in range(trainPredWin)
#            trainingDataEEG2[i,j,:]=[trainingDataEEG[i,j,:] trainingDataEEG[i,j+1,:]]
#        i+=1
    
    '''Validation Set'''
   # j=0
    develDataEEG=np.zeros(((numBlocks-287),Window10s,numChans+1))
    develDataAudioA=np.zeros(((numBlocks-287),Window10s,1))
    develDataAudioU=np.zeros(((numBlocks-287),Window10s,1))
    
            
   # for block1 in range(numBlocks):
    #        if block1 in leaveValidat:      #leaveBlocks[:]==block:
     #           develDataEEG[j,:,:]=eegData[block1,:,:]
      #          develDataAudioA[j,:,0]=targetAudio[block1,:,b] 
       #         develDataAudioU[j,:,0]=targetAudio[block1,:,a] 
        #        j+=1
            
            
    
    '''Testing Set'''
    #k=0
    testDataEEG=np.zeros(((numBlocks-287),Window10s,numChans+1))
    testDataAudioA=np.zeros(((numBlocks-287),Window10s,1))
    testDataAudioU=np.zeros(((numBlocks-287),Window10s,1))
            
    #for block2 in range(numBlocks):
           # if block2 in leaveTest:      #leaveBlocks[:]==block:
                #testDataEEG[k,:,:]=eegData[block2,:,:]
                #testDataAudioA[k,:,0]=targetAudio[block2,:,b] 
                #testDataAudioU[k,:,0]=targetAudio[block2,:,a] 
                #k+=1
            
    
    StartLagTrain=np.zeros(((numBlocks-2),Window10s,numChans+1))
    EndLagTrain=np.zeros(((numBlocks-2),Window10s,numChans+1))
    StartLagDevel=np.zeros(((numBlocks-287),Window10s,numChans+1))
    EndLagDevel=np.zeros(((numBlocks-287),Window10s,numChans+1))
    StartLagTest=np.zeros(((numBlocks-287),Window10s,numChans+1))
    EndLagTest=np.zeros(((numBlocks-287),Window10s,numChans+1)) 
    results1=np.zeros((5))*np.nan



    '''Example for using np.pad'''
#A = np.array([[1,2],[3,4]])
#np.pad(A, ((1,2),(2,1)), 'constant')

#array([[0, 0, 0, 0, 0],           # 1 zero padded to the top
#       [0, 0, 1, 2, 0],           # 2 zeros padded to the bottom
#       [0, 0, 3, 4, 0],           # 2 zeros padded to the left
#       [0, 0, 0, 0, 0],           # 1 zero padded to the right
#       [0, 0, 0, 0, 0]])

    #corrCoefA=np.zeros((len(LAGS15),72, 1))
    #corrCoefU=np.zeros((len(LAGS15),72, 1))
    predicted = np.zeros((len(LAGS15),numBlocks, 2))
    tested = np.zeros((len(LAGS15),numBlocks, 2))
    
    Acc2=np.zeros((len(LAGS15), 1))

        
        
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
                #trainingDataEEG1= trainingDataEEG.reshape((trainingDataEEG.shape[0]*trainingDataEEG.shape[1]), trainingDataEEG.shape[2])
                blockE = int(np.floor(block/6))
                trainingDataAudioA[i,:,0]=targetAudio[block,:,b[blockE]]
                trainingDataAudioA1= trainingDataAudioA.reshape((trainingDataAudioA.shape[0]*trainingDataAudioA.shape[1]), trainingDataAudioA.shape[2])
                trainingDataAudioU[i,:,0]=targetAudio[block,:,a[blockE]]
                trainingDataAudioU1= trainingDataAudioU.reshape((trainingDataAudioU.shape[0]*trainingDataAudioU.shape[1]), trainingDataAudioU.shape[2])
                i+=1
            k=0
            develDataEEG[:,:,:]=eegData[leaveValidat11,:,:]
            blockV = int(np.floor(leaveValidat11/6))
            #develDataEEG1= develDataEEG.reshape((develDataEEG.shape[0]*develDataEEG.shape[1]), develDataEEG.shape[2])
            develDataAudioA[:,:,0]=targetAudio[leaveValidat11,:,b[blockV]]
            develDataAudioA1= develDataAudioA.reshape((develDataAudioA.shape[0]*develDataAudioA.shape[1]), develDataAudioA.shape[2])
            develDataAudioU[:,:,0]=targetAudio[leaveValidat11,:,a[blockV]] 
            develDataAudioU1= develDataAudioU.reshape((develDataAudioU.shape[0]*develDataAudioU.shape[1]), develDataAudioU.shape[2])
    
            testDataEEG[:,:,:]=eegData[leaveTest11,:,:]
            blockT = int(np.floor(leaveTest11/6))
            #testDataEEG1= testDataEEG.reshape((testDataEEG.shape[0]*testDataEEG.shape[1]), testDataEEG.shape[2])
            testDataAudioA[:,:,0]=targetAudio[leaveTest11,:,b[blockT]]
            testDataAudioA1= testDataAudioA.reshape((testDataAudioA.shape[0]*testDataAudioA.shape[1]), testDataAudioA.shape[2])
            testDataAudioU[:,:,0]=targetAudio[leaveTest11,:,a[blockT]]
            testDataAudioU1= testDataAudioU.reshape((testDataAudioU.shape[0]*testDataAudioU.shape[1]), testDataAudioU.shape[2])
    
            StartLagDevel[k,:,:]= np.pad(develDataEEG[k,:,:], ((0, start), (0, 0)), mode='constant')[start:, :]
            EndLagDevel[k,:,:]=np.pad(develDataEEG[k,:,:], ((0, fin), (0, 0)), mode='constant')[fin:, :]
            DevelDataEEGLagged=np.concatenate([StartLagDevel, EndLagDevel], axis=2)
            DevelDataEEGLagged1= DevelDataEEGLagged.reshape((DevelDataEEGLagged.shape[0]*DevelDataEEGLagged.shape[1]), DevelDataEEGLagged.shape[2])
            StartLagTest[k,:,:]= np.pad(testDataEEG[k,:,:], ((0, start), (0, 0)), mode='constant')[start:, :]
            EndLagTest[k,:,:]=np.pad(testDataEEG[k,:,:], ((0, fin), (0, 0)), mode='constant')[fin:, :]
            TestDataEEGLagged=np.concatenate([StartLagTest, EndLagTest], axis=2)
            TestDataEEGLagged1= TestDataEEGLagged.reshape((TestDataEEGLagged.shape[0]*TestDataEEGLagged.shape[1]), TestDataEEGLagged.shape[2])
    
            
            for block in range(numBlocks-2):
                StartLagTrain[block,:,:] = np.pad(trainingDataEEG[block,:,:], ((0, start), (0, 0)), mode='constant')[start:, :]
                EndLagTrain[block,:,:]   = np.pad(trainingDataEEG[block,:,:], ((0, fin), (0, 0)), mode='constant')[fin:, :]
                TrainingDataEEGLagged=np.concatenate([StartLagTrain, EndLagTrain], axis=2)
            TrainingDataEEGLagged1= TrainingDataEEGLagged.reshape((TrainingDataEEGLagged.shape[0]*TrainingDataEEGLagged.shape[1]), TrainingDataEEGLagged.shape[2])
    
            '''Early Stopping to get to the point in which the loss on the Development set does not decrease anymore'''
            
            Model=createModel()
            tempModelName=workingDir+'/model/RevSingle.hdf5'
            checkLow = ModelCheckpoint(filepath=tempModelName, verbose=0, save_best_only=True,mode='min',monitor='val_loss')            
            early = EarlyStopping(monitor='val_loss',patience=10, mode='min')
            
            trainingDataAudio = np.concatenate((trainingDataAudioA[:,:,:],trainingDataAudioU[:,:,:]),axis=2)
            develDataAudio = np.concatenate((develDataAudioA[:,:,:],develDataAudioU[:,:,:]),axis=2)
            testDataAudio = np.concatenate((testDataAudioA[:,:,:],testDataAudioU[:,:,:]),axis=2)
            '''Training Model'''
                   # explore parameter batch_size between 8-19-38
            #print(Model.summary())
            Model.fit(TrainingDataEEGLagged[:,:,:],trainingDataAudio[:,:,:],batch_size=2,epochs=150,verbose=1,callbacks=[early,checkLow],validation_data=(DevelDataEEGLagged[:,:,:],develDataAudio[:,:,:]))
            #Model.fit(trainingDataEEG[:,:,:],trainingDataAudioA[:,:,:],batch_size=38,epochs=300,verbose=1,validation_data=(develDataEEG[:,:,:],develDataAudioA[:,:,:]))
            Model.load_weights(tempModelName)
        
        
            '''Prediction'''
        
            predictionA=Model.predict(TestDataEEGLagged[:,:,:])    #[:,:,:] #Are we using LAG also for Test, in linear model was so
            
           # corrCoefA=np.zeros(((len(numBlocks)), 1))
            #corrCoefU=np.zeros((numBlocks, 1))
            predicted[end_lagi,blockCV,:] = np.mean(predictionA,axis=1)
            tested[end_lagi,blockCV,:] = np.mean(testDataAudio,axis=1)
            
            
            
        Acc2[end_lagi,:] = 0

io.savemat(workingDir+'/Results/RevMulti_'+str(Subject)+'_CI8_decission_50sv2.0EnvSum.mat',{'predicted'+str(Subject):predicted, 'Growntrouth'+str(Subject):tested, 'Acc2'+str(Subject):Acc2})    #'corrA':np.squeeze(corrA[str(aWindow)]),'corrU':np.squeeze(corrU[str(aWindow)])})
           
            
            
           # corrA1=dict()
           # corrU1=dict()
           # corrA1[str(end_lag)]=np.zeros((int(trainPredWin),1))*np.nan
           # corrU1[str(end_lag)]=np.zeros((int(trainPredWin),1))*np.nan
            
        #corrCoefA=np.zeros(((len(leaveTest)), 1))
       # corrCoefU=np.zeros(((len(leaveTest)), 1))
           # for block in range(len(leaveTest)):
                #corrA1[str(end_lag)][part,samp,0]=np_logcosh(testDataI[part,samp:samp+trainPredWin,chans],predictionA[part,samp:samp+trainPredWin,chans])
                #corrU1[str(end_lag)][part,samp,0]=np_logcosh(testDataI[part,samp:samp+trainPredWin,chans],predictionU[part,samp:samp+trainPredWin,chans]) 
                #corrCoefA[block,:]  = 1-correlation_coefficient_loss_np(testDataAudioA[block,:,:],predictionA[block,:,:])
                #corrCoefU[block,:]  = 1-correlation_coefficient_loss_np(testDataAudioU[block,:,:],predictionA[block,:,:])
            
            #results1[end_lagi,0]=end_lag
            #results1[end_lagi,1]=np.mean(corrCoefA>corrCoefU)
            #results1[end_lagi,2]=np.mean(corrCoefA)
            #results1[end_lagi,3]=np.mean(corrCoefU)
            #results1[end_lagi,4]=Model.count_params()   

#    io.savemat(workingDir+'/Results/RevMulti_'+str(Subject)+'_NHcEEGrid.mat',{'results1'+str(Subject):results1, 'Acc2'+str(Subject):Acc2})    #'corrA':np.squeeze(corrA[str(aWindow)]),'corrU':np.squeeze(corrU[str(aWindow)])})


          #''' Calculate auditory attention decoding by comparison of predicted EEG in botch cases    
    
#            corrA=dict()
#            corrU=dict()
#            analysisWindows=[fs*30,fs*10,fs*5,fs*3,fs*2,fs*1,int(fs*0.5)]
    
    
#            corrA1=np.zeros(((len(leaveTest)), 1))
#            corrU1=np.zeros(((len(leaveTest)), 1))
    
#            corrCoefA=np.zeros(((len(leaveTest)), 1))
#            corrCoefU=np.zeros(((len(leaveTest)), 1))
    
#                for block in range(len(leaveTest)):
        #using logcosh
#                    corrA1[block,:]=np_logcosh(testDataAudioA[block,:,:],predictionA[block,:,:])
#                    corrU1[block,:]=np_logcosh(testDataAudioU[block,:,:],predictionA[block,:,:]) 
        
        
#                    corrCoefA[block,:]  = 1-correlation_coefficient_loss_np(testDataAudioA[block,:,:],predictionA[block,:,:])
#                   corrCoefU[block,:]  = 1-correlation_coefficient_loss_np(testDataAudioU[block,:,:],predictionA[block,:,:])
       
        
        #a = np.array([testDataAudioA[block,:,:], predictionA[block,:,:]])
        #corrCoefA[block,:] = np.corrcoef(a[:,:,0])[0,1];
        
        #b = np.array([testDataAudioU[block,:,:], predictionA[block,:,:]])
        #corrCoefU[block,:] = np.corrcoef(b[:,:,0])[0,1];
        
    
        #corrCoefA[block,:] = np.corrcoef(testDataAudioA[block,:,:],predictionA[block,:,:])
        #corrCoefU[block,:] = np.corrcoef(testDataAudioU[block,:,:],predictionA[block,:,:])
        #using corrcoef
        #corrCoefA[block,:] = np.corrcoef(np.reshape(testDataAudioA[block,:,:],(3840,1)),np.reshape(predictionA[block,:,:],(3840,1)));
        #corrCoefU[block,:] = np.corrcoef(np.reshape(testDataAudioU[block,:,:],(3840,1)),np.reshape(predictionA[block,:,:],(3840,1)));
    
#                    Acc  = np.mean(corrU1<corrA1)
#                    Acc2 = np.mean(corrCoefU<corrCoefA)         
    
    
#    results=np.zeros((len(analysisWindows),4))*np.nan
#    for win in range(len(analysisWindows)):
#        aWindow=analysisWindows[win]
#        corrA[str(aWindow)]=np.zeros((5,int((trainPredWin-aWindow)),1))*np.nan
#        corrU[str(aWindow)]=np.zeros((5,int((trainPredWin-aWindow)),1))*np.nan
#        for block in range(len(leaveTest)):
#            for samp in range(0,int((trainPredWin-aWindow))):
#                corrA[str(aWindow)][block,samp,0]=np_logcosh(testDataAudioA[block,samp:samp+aWindow,:],predictionA[block,samp:samp+aWindow,:])
#                corrU[str(aWindow)][block,samp,0]=np_logcosh(testDataAudioU[block,samp:samp+aWindow,:],predictionA[block,samp:samp+aWindow,:])            
#        results[win,0]=np.mean(corrU[str(aWindow)]<corrA[str(aWindow)])
#        results[win,1]=np.mean(corrA[str(aWindow)])
#        results[win,2]=np.mean(corrU[str(aWindow)])
#        results[win,3]=Model.count_params()   '''
    
    
    
    
#    io.savemat(workingDir+'/Results/RevMulti_'+str(Subject)+'_NHcEEGrid.mat',{'results'+str(Subject):results, 'Acc'+str(Subject):Acc})    #'corrA':np.squeeze(corrA[str(aWindow)]),'corrU':np.squeeze(corrU[str(aWindow)])})
     
    
    
    




