# SelectiveAttentionDNN
This repository demonstrates python codes that use keras and tensorflow to decode selective attention from electroencephalography (EEG).
For more details on the algorithms refer to the IEEE ICASSP 2020 paper: Waldo Nogueira, Hanna Dolhopiatenko, TOWARDS DECODING SELECTIVE ATTENTION FROM SINGLE-TRIAL EEG DATA IN COCHLEAR IMPLANT USERS BASED ON DEEP NEURAL NETWORKS. Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054021

Two selective attention decoder architectures were investigated: 
    1) Decoder based on reconstruction of speech ('FCNetwork1min.py', 'FCNetwork10s', 'CNNNetwork10s')
    2) Decoder based on decision on locus of attention ('DecissionEEG.py', 'DecissionEnvSpeech.py' and 'DecissionEnvSpeechSum.py').

Original audio (48 min, sampling rate=64 Hz) and one example of EEG Dataset (48 min, sampling rate=64 Hz) are provided. 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Short Description of provided files: 

• 'LoadData'/'LoadData2' - modules which load data.

RECONTRUCTION:

• 'FCNetwork1min.py' - Fully-connected deep neural network. 46 min of EEG is used for training, one minute for validation and one for testing. Cross-validation 'leave-one-out' is implemented with 5 folds. 

• 'FCNetwork10s.py' - Fully-connected deep neural network. 47min40s of EEG is used for training, 10 seconds for validation and 10 seconds for testing. Cross-validation 'leave-one-out' is implemented with 72 folds.

• 'CNNNetwork10s.py' - Convolutional deep neural network. 47min40s of EEG is used for training, 10 seconds for validation and 10 seconds for testing. Cross-validation 'leave-one-out' is implemented with 72 folds.

DECISION:

• 'DecissionEEG.py' - Fully-connected deep neural network.  47min40s of EEG is used for training, 10 seconds for validation and 10 seconds for testing). Cross-validation 'leave-one-out' is implemented with 72 folds. It only uses the lagged EEG as input without
speech envelopes.

• 'DecissionEnvSpeech.py'-Fully-connected deep neural network.  47min40s of EEG is used for training, 10 seconds for validation and 10 seconds for testing). Cross-validation 'leave-one-out' is implemented with 72 folds. Left and right stimulus envelopes were mixed in a single channel that was appended to the lagged EEG data matrix.

• 'DecissionEnvSpeechSum.py' Fully-connected deep neural network.  47min40s of EEG is used for training, 10s second for validation and 10 seconds for testing). Cross-validation 'leave-one-out' is implemented with 72 folds. Left and the right stimulus envelopes were appended to the lagged EEG data matrix
