# SelectiveAttentionDNN
This repository demonstrates python codes which uses keras and tensorflow to decode selective attention from electroencephalography (EEG).
For more details, read: SINGLE-TRIAL EEG DATA IN COCHLEAR IMPLANT USERS BASED ON DEEP NEURAL NETWORKS, Waldo Nogueira, Hanna Dolhopiatenko. Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054021

Two types of decoders were investigated: decoder based on reconstruction of speech ('FCNetwork1min.py', 'FCNetwork10s', 'CNNNetwork10s') and decoder based on decision on locus of attention ('DecissionEEG.py', 'DecissionEnvSpeech.py' and 'DecissionEnvSpeechSum.py').

Original audio (48min, sampling rate=64Hz) and one example of EEG Dataset (48min, sampling rate=64Hz) are provided. 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Short Description of provided files: 

• 'LoadData'/'LoadData2' - modules which load data.

RECONTRUCTION:

• 'FCNetwork1min.py' - Fully-connected deep neural network. 46 min of signals is used for training, one minute for validation and one for testing. Cross-validation 'leave-one-out' is implemented with 5 steps. 

• 'FCNetwork10s.py' - Fully-connected deep neural network. 47min40s of signals used for training, 10second for validation and 10second for testing. Cross-validation 'leave-one-out' is implemented with 72 steps.

• 'CNNNetwork10s.py' - Convolutional deep neural network. 47min40s of signals used for training, 10second for validation and 10second for testing. Cross-validation 'leave-one-out' is implemented with 72 steps.

DECISION:

• 'DecissionEEG.py' - Fully-connected deep neural network.  47min40s of signals used for training, 10second for validation and 10second for testing). Cross-validation 'leave-one-out' is implemented with 72 steps. Only uses the lagged EEG as input without
speech envelopes.

• 'DecissionEnvSpeech.py'-Fully-connected deep neural network.  47min40s of signals used for training, 10second for validation and 10second for testing). Cross-validation 'leave-one-out' is implemented with 72 steps. Left and right stimulus envelopes were mixed in a single channel that was appended to the lagged EEG data matrix.

• 'DecissionEnvSpeechSum.py' Fully-connected deep neural network.  47min40s of signals used for training, 10second for validation and 10second for testing). Cross-validation 'leave-one-out' is implemented with 72 steps. Left and the right stimulus envelope were appended to the lagged EEG data matrix
