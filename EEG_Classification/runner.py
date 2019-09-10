#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:21:56 2017

@author: camasa
"""

import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


from glob import glob

from scipy.signal import butter, lfilter, convolve, boxcar
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

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

def run(subjectsNum,classifier, subFile):
    print 'hi as we go'

    subjects = range(1,subjectsNum)
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
    submission_file = subFile
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
        feattr = np.log(feattr[:,0:feat.shape[1]])
        
        # training labels
        # they are stored in the 6 last channels of the MNE raw object
        labels = raw._data[32:]
        
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
        featte = np.log(featte[:,0:feat.shape[1]])
        
        tru = raw._data[32:].T
        
    #    feattrNorm = normalize(feattr)
    #    featteNorm = normalize(featte)
    #    
    #    ctr, rtr = feattr.shape
    #    ctt, rtt = featte.shape
    #    print ctr, rtr
    #    print ctt, rtt
        ############## Train classifiers ########################################
        if classifier == 'LR':
            lr = LogisticRegression()
            pred = np.empty((len(ids),6))
            for i in range(6):
                print('LR Train subject %d, class %s' % (subject, cols[i]))
                lr.fit(feattr[:,::subsample].T,labels[i,::subsample])
                pred[:,i] = lr.predict_proba(featte.T)[:,1]
        
#            pred_tot.append(pred)
#            true = np.r_[true,tru]
        elif classifier == 'MLP':
            mlp = MLPClassifier(solver='sgd',learning_rate = 'adaptive', hidden_layer_sizes=(1000, ),learning_rate_init =0.0005, max_iter = 1000000,shuffle = False)
            pred = np.empty((len(ids),6))
            for i in range(6):
                print('MLP Train subject %d, class %s' % (subject, cols[i]))
                mlp.fit(feattr[:,::subsample].T,labels[i,::subsample])
                pred[:,i] = mlp.predict_proba(featte.T)[:,1]
            
#            pred_tot.append(pred)
#            true = np.r_[true,tru]
            
#        elif classifier == 'SVC':
#            svr = SVC()
#            pred = np.empty((len(ids),6))
#            for i in range(6):
#                print('SVC Train subject %d, class %s' % (subject, cols[i]))
#                svr.fit(feattr[:,::subsample].T,labels[i,::subsample])
#                pred[:,i] = svr.predict(featte.T)[:,1]
        elif classifier == 'RF':
            rf = RandomForestClassifier(n_estimators=100,max_features ='auto',n_jobs=-1)
            pred = np.empty((len(ids),6))
            for i in range(6):
                print('RF Train subject %d, class %s' % (subject, cols[i]))
                rf.fit(feattr[:,::subsample].T,labels[i,::subsample])
                pred[:,i] = rf.predict_proba(featte.T)[:,1]
                #score = 
        elif classifier == 'SVC':
            svc = LinearSVC()  
            pred = np.empty((len(ids),6))
            for i in range(6):
                print('RF Train subject %d, class %s' % (subject, cols[i]))
                svc.fit(feattr[:,::subsample].T,labels[i,::subsample])
                pred[:,i] = svc.predict(featte.T)#[:,1]

            
        pred_tot.append(pred)
        true = np.r_[true,tru]
            
    data  = np.concatenate(pred_tot)
#    score = mlp.score(true,data)
#    print score
#    auc = roc_auc_score(true,data)
#    print auc
#    fpr = dict()
#    tpr = dict()
#    thresholds = dict()
#    roc_auc = dict()
#    
#    for i in range(6):
#        fpr, tpr, _ = roc_curve(true[:,i], data[:,i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    for j in range(6):
#        plt.figure()
#        lw = i
#        plt.plot(fpr[j], tpr[j], color='darkorange',
#                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[j])
#        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title('Receiver operating characteristic example')
#        plt.legend(loc="lower right")
#        plt.savefig('/Users/camasa/Desktop/roc%d.eps' % (j),format='eps')
#        plt.show()
    
        
        
        
        
    # create pandas object for sbmission
    submission = pd.DataFrame(index=np.concatenate(ids_tot),
                              columns=cols,
                              data=np.concatenate(pred_tot))
    
    # write file
    submission.to_csv(submission_file,index_label='id',float_format='%.5f')
    
    trueLabels = pd.DataFrame(index=np.concatenate(ids_tot),
                              columns=cols,
                              data = true)
    trueLabels.to_csv('TrueLabels.csv',index_label='id',float_format='%.5f')
    
    
    
run(13,'SVC','SVCFinal.csv')


    
    
    
    
    
    
    
    
    
    