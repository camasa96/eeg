#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:50:03 2017

@author: camasa
"""

from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')
#__reset__()

from scipy.signal import butter, lfilter
import preprocessing as pp
#import filterBank    as fb
import numpy as np




freqs_pairs = [[0.5], [1], [2], [3], [4], [5], [7], [9], [15],[30]]
X_tot = None
for freqs in freqs_pairs:
    if len(freqs) == 1:
        b, a = butter(5, freqs[0] / 250.0, btype='lowpass')
    else:
        if freqs[1] - freqs[0] < 3:
            b, a = butter(3, np.array(freqs) / 250.0, btype='bandpass')
        else:
            b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')


[data_train, labels_train, data_test, labels_test]=pp.load_raw_data(1,test=False)





#__reset__()
