que #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:28:49 2017

@author: camasa
"""
from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')
#__reset__()

import numpy as np
import pandas as pd
import mne


def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

data,labels =  prepare_data_train('/Users/camasa/Documents/University/Grad/DM_Pro/train/subj1_series2_data.csv')


data   =np.asarray(data.astype(float))
labels =np.asarray(labels.astype(float))


# Some information about the channels
ch_names = ['Fp1', 'Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7',
            'C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz',
            'P4','P8','PO9','O1','Oz','O2','PO10'] 

# Sampling rate 
sfreq = 500  # Hz
# Create the info structure needed by MNE
info = mne.create_info(ch_names, sfreq, ch_types=  'eeg')

# Finally, create the Raw object
raw = mne.io.RawArray(np.transpose(data), info)
#data, times = raw[:30, int(sfreq * 1):int(sfreq * 3)]
# Plot it!
#plt.plot(times, data.T)
#data.plot()

#__reset__()