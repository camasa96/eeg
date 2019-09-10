# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:00:37 2015

@author: alexandrebarachant

Beat the benchmark with CSP and Logisitic regression.

General Idea :

The goal of this challenge is to detect events related to hand movements. Hand 
movements are caracterized by change in signal power in the mu (~10Hz) and beta
(~20Hz) frequency band over the sensorimotor cortex. CSP spatial filters are
trained to enhance signal comming from this brain area, instantaneous power is
extracted and smoothed, and then feeded into a logisitic regression.

Preprocessing :

Signal are bandpass-filtered between 7 and 30 Hz to catch most of the signal of
interest. 4 CSP spatial filters are then applied to the signal, resutlting to
4 new time series.  In order to train CSP spatial filters, EEG are epoched 
using a window of 2 second before and after the event 'HandStart'. CSP training
needs two classes. the epochs after Replace event are assumed to contain 
patterns corresponding to hand movement, and epochs before are assumed to 
contain resting state.

Feature extraction :

Preprocessing is applied, spatialy filtered signal are the rectified and 
convolved with a 0.5 second rectangular window for smoothing. Then a logarithm
is applied. the resutl is a vector of dimention 4 for each time sample.

Classification :

For each of the 6 event type, a logistic regression is trained. For training 
only, features are downsampled in oder to speed up the process. Prediction are
the probailities of the logistic regression.

"""
from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')
#__reset__()


import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import  libsvm,LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from glob import glob

from scipy.signal import butter, lfilter, convolve, boxcar
from joblib import Parallel, delayed
print 'helf '

def creat_mne_raw_object(fname,read_events=True):
    """Create a mne raw instance from csv file"""
    # Read EEG file
    data = pd.read_csv(fname)
    
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T
    
    if read_events:
        # events file
        ev_fname = fname.replace('_data','_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T
        
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))
        
    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    #info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    
    return raw

subjects = range(1,2)
ids_tot = []
pred_tot = []
true = np.empty((0,6))

# design a butterworth bandpass filter 
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 4

# convolution
# window for smoothing features
nwin = 250

# training subsample
subsample = 10

# submission file
submission_file = 'MLP.csv'
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

for subject in subjects:
    epochs_tot = []
    y = []

    ################ READ DATA ################################################
    fnames =  glob('/Users/camasa/Documents/University/Grad/DM_Pro/train1/subj%d_series*_data.csv' % (subject))
    
    # read and concatenate all the files
    raw = concatenate_raws([creat_mne_raw_object(fname) for fname in fnames])
       
    # pick eeg signal
    picks = pick_types(raw.info,eeg=True)
    
    # Filter data for alpha frequency and beta band
    # Note that MNE implement a zero phase (filtfilt) filtering not compatible
    # with the rule of future data.
    # Here we use left filter compatible with this constraint. 
    # The function parallelized for speeding up the script
    raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raw._data[i]) for i in picks))
    
    ################ CSP Filters training #####################################
    # get event posision corresponding to HandStart
    events = find_events(raw,stim_channel='HandStart', verbose=False)
    # epochs signal for 2 second after the event
    epochs = Epochs(raw, events, {'during' : 1}, 0, 2, proj=False,
                    picks=picks, baseline=None, preload=True,
                    verbose=False)
    
    epochs_tot.append(epochs)
    y.extend([1]*len(epochs))
    
    # epochs signal for 2 second before the event, this correspond to the 
    # rest period.
    epochs_rest = Epochs(raw, events, {'before' : 1}, -2, 0, proj=False,
                    picks=picks, baseline=None, preload=True,
                     verbose=False)
    
    # Workaround to be able to concatenate epochs with MNE
    epochs_rest.times = epochs.times
    
    y.extend([-1]*len(epochs_rest))
    epochs_tot.append(epochs_rest)
        
    # Concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)
    
    # get data 
    X = epochs.get_data()
    y = np.array(y)
    
    # train CSP
    csp = CSP(n_components=nfilters, reg='ledoit_wolf')
    csp.fit(X,y)
    
    ################ Create Training Features #################################
    # apply csp filters and rectify signal
    feat = np.dot(csp.filters_[0:nfilters],raw._data[picks])**2
    
    # smoothing by convolution with a rectangle window    
    feattr = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[i],boxcar(nwin),'full') for i in range(nfilters)))
    feattr = np.log(feattr[:,0:feat.shape[1]],order='F')
    
    # training labels
    # they are stored in the 6 last channels of the MNE raw object
    labels = raw._data[32:]
    labels = np.array(labels, order = 'F')
    
    ################ Create test Features #####################################
    # read test data 
    fnames =  glob('/Users/camasa/Documents/University/Grad/DM_Pro/test1val/subj%d_series*_data.csv' % (subject))
    raw = concatenate_raws([creat_mne_raw_object(fname) for fname in fnames])
    raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raw._data[i]) for i in picks))
    
    # read ids
    ids = np.concatenate([np.array(pd.read_csv(fname)['id']) for fname in fnames])
    ids_tot.append(ids)
    
    # apply preprocessing on test data
    feat = np.dot(csp.filters_[0:nfilters],raw._data[picks])**2
    featte = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[i],boxcar(nwin),'full') for i in range(nfilters)))
    featte = np.log(featte[:,0:feat.shape[1]],order='F')
    
    tru = raw._data[32:].T
    
    ################ Standardize features #####################################
#    scaler = StandardScaler()
#    featte = scaler.fit_transform(featte)
#    feattr = scaler.fit_transform(feattr)
#    
#    
#    feattrNorm = normalize(feattr)
#    featteNorm = normalize(featte)
#    
#    ctr, rtr = feattr.shape
#    ctt, rtt = featte.shape
#    print ctr, rtr
#    print ctt, rtt
    ############## Train classifiers ########################################
#    lr = LogisticRegression()
#    pred = np.empty((len(ids),6))
#    for i in range(6):
#        print('Train subject %d, class %s' % (subject, cols[i]))
#        lr.fit(feattr[:,::subsample].T,labels[i,::subsample])
#        pred[:,i] = lr.predict_proba(featte.T)[:,1]
#    
#    pred_tot.append(pred)
#    true = np.r_[true,tru]
    
    
    
    
#    mlp = MLPClassifier(solver='lbfgs')
#    pred = np.empty((len(ids),6))
#    for i in range(6):
#        print('Train subject %d, class %s' % (subject, cols[i]))
#        mlp.fit(feattr[:,::subsample].T,labels[i,::subsample])
#        pred[:,i] = mlp.predict_proba(featte.T)[:,1]
##        
#    pred_tot.append(pred)
#    mlp = MLPClassifier(solver='sgd',learning_rate = 'adaptive', hidden_layer_sizes=(500,2 ),learning_rate_init =0.0001, max_iter = 1000000,shuffle = False)
#    pred = np.empty((len(ids),6))
#    for i in range(6):
#        print('MLP Train subject %d, class %s' % (subject, cols[i]))
#        mlp.fit(feattr[:,::subsample].T,labels[i,::subsample])
#        pred[:,i] = mlp.predict_proba(featte.T)[:,1]
    
#    svr = libsvm()
#    pred = np.empty((len(ids),6))
#    for i in range(6):
#        print('Train subject %d, class %s' % (subject, cols[i]))
#        s = libsvm.fit(np.ascontiguousarray(feattr[:,::subsample].T),np.ascontiguousarray(labels[i,::subsample]),gamma=50)
#        support = s[0]
#        SV = s[1]
#        nSV = s[2]
#        sv_coef = s[3]
#        intercept = s[4]
#        pred[:,i] = libsvm.predict_proba(np.ascontiguousarray(featte.T),support,SV,nSV,sv_coef,intercept)[:,1]
#

    svc = LinearSVC(multi_class="crammer_singer")
    pred = np.empty((len(ids),6))
    for i in range(6):
        print('Train subject %d, class %s' % (subject, cols[i]))
        svc.fit(feattr[:,::subsample].T,labels[i,::subsample])
        pred[:,i] = svc.predict(featte.T)#[:,1]
    
    pred_tot.append(pred)
    true = np.r_[true,tru]
    
data  = np.concatenate(pred_tot)
#true = np.r_[true,tru]
auc = roc_auc_score(true,data)
print auc
    
    
    
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.5f')

