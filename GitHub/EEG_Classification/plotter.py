#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 03:50:49 2017

@author: camasa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

y_test = pd.read_csv('/Users/camasa/Documents/University/Grad/DM_Pro/TrueLabels.csv')
y_score = pd.read_csv('/Users/camasa/Documents/University/Grad/DM_Pro/SVCFinal.csv')

y_test = y_test.loc[:,'HandStart':'BothReleased']
y_score = y_score.loc[:,'HandStart':'BothReleased']

y_test = np.asarray(y_test.astype(float))
y_score = np.asarray(y_score.astype(float))


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i].T, y_score[:, i].T)
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['b','g','r','c','m','k'])
for i, color in zip(range(6), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of multiple classes Random Forest')
plt.legend(loc="lower right")
plt.savefig('/Users/camasa/Documents/University/Grad/DM_Pro/svc.eps',format='eps')
plt.show()
 
