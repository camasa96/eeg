## Classification of EEG Signals
# Prepared for Data Mining Grad Course
# Fall 2017

This project is my take on the https://www.kaggle.com/c/grasp-and-lift-eeg-detection.

Given: EEGG recordings of subject performing grasp-and-lift trials.

Objectives: Identify hand motions from EEG recordings

Approach: Filter data using spatial filters. Feed resulting features to Logistic Regresion, MLP and Random forest models.

Results: Models assessed on ROC and AUC. Linear regression model return higheset AUC.


Used https://www.kaggle.com/alexandrebarachant/common-spatial-pattern-with-mne and https://www.kaggle.com/elenacuoco/simple-grasp-with-sklearn as starting point 
