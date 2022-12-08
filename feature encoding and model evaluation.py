# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 02:31:30 2021

@author: HXu8
"""
import os, re, sys
import numpy as np
import pandas as pd
from math import log
import random
import scipy.io

from keras.layers import *
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.initializers import RandomUniform, RandomNormal, glorot_uniform, glorot_normal
from keras.regularizers import l1, l2, l1_l2
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib
import matplotlib.pyplot as plt
from numpy import interp
np.random.seed(1234)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def read_blosum(path,one_hot):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    f = open(path,"r")
    blosum = []
    if one_hot ==0: #(blosum 50)
       for line in f:
           blosum.append([(float(i))/10 for i in re.split("\t",line)])
    else:
        for line in f: #(one-hot)
           blosum.append([float(i) for i in re.split("\t",line)])
    f.close()
    return blosum

path_dict = "your path"
blosum_matrix = read_blosum(path_dict + 'blosum50.txt', 0)

pseq_dict = np.load(path_dict + 'pseq_dict_all.npy', allow_pickle = True).item()
pseq_dict_blosum_matrix = pseudo_seq(pseq_dict, blosum_matrix)

def creat_your_model(training_pep, training_mhc):
    xxx
    return model

def mhc_peptide_pair(your_file_path, pseq_dict_matrix, blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
    data_dict = {}
    pep_length = [8,9,10,11,12,13,14,15]
    f = open(path,"r")
    for line in f:
        info = re.split("\t",line)#Retrive information from a tab-delimited line
        allele = info[0].strip()
        if allele in pseq_dict.keys():
            affinity = int(info[1].strip()) #Retrive lable information 
            pep = info[-1].strip() #Retrive ligand information 
            
            if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
                print('Illegal peptides')
                continue   
            if len(pep) not in pep_length:
                print('Illegal peptides')
                continue 
                
            pep_blosum = []#Encoded peptide seuqence
            for residue_index in range(15):
                #Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
                #If the peptide is shorter than 12 residues, the remaining positions on
                #the rightare filled will zero-padding
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(15):
                #Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
                #If the peptide is shorter than 12 residues, the remaining positions on
                #the left are filled will zero-padding
                if 15 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20)) 
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 15 + residue_index]]])

            new_data = [pep_blosum, pseq_dict_matrix[allele], affinity]
            
            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)
                
    return data_dict

def main_model_training_cvs():
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    #Load the blosum matrix for encoding
    folder = "your model save files"
    if not os.path.isdir(folder):
        os.makedirs(folder)  

    path_train = "your train datasets"
    data_train_dict = mhc_peptide_pair(path_train, pseq_dict_blosum_matrix, blosum_matrix)    
      
    training_data = []
    for allele in data_train_dict.keys():
        allele_data = data_train_dict[allele]
        random.shuffle(allele_data)
        allele_data = np.array(allele_data)
        training_data.extend(allele_data)
    
    [all_pep, all_mhc, all_target] = [[i[j] for i in training_data] for j in range(3)]
    all_pep = np.array(all_pep)
    all_mhc = np.array(all_mhc)
    all_target = np.array(all_target)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)    
    allprobas_=np.array([]) 
    allylable=np.array([])
    #####train
    for i, (train, test) in enumerate(kfold.split(all_pep, all_target)):
        training_pep = all_pep[train]
        training_mhc = all_mhc[train]
        training_target = all_target[train]
        
        validation_pep = all_pep[test]
        validation_mhc = all_mhc[test]
        validation_target = all_target[test]

        mc = ModelCheckpoint(folder + '/model_%s.h5' % str(i), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        model = creat_your_model(training_pep, training_mhc)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        model.fit([training_pep,training_mhc], 
                        training_target,
                        batch_size=512,
                        epochs = 100,
                        shuffle=True,
                        callbacks=[es, mc],
                        validation_data=([validation_pep,validation_mhc], validation_target),
                        verbose=1)
        
        saved_model = load_model(folder + '/model_%s.h5' % str(i))
        probas_ = saved_model.predict([np.array(validation_pep),np.array(validation_mhc)])
        allprobas_ = np.append(allprobas_, probas_)           
        allylable = np.append(allylable, validation_target)
        del model

    font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16}
    figsize=6.2, 6.2

    ########ROC_figure
    figure1, ax1 = plt.subplots(figsize=figsize)
    ax1.tick_params(labelsize=18)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  

    fpr, tpr, thresholds = roc_curve(allylable, allprobas_)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    
    ax1.plot(fpr, tpr, color='b',
        label=r'Mean ROC (AUC = %0.4f)' % (roc_auc),
        lw=2, alpha=.8)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate', font1)
    ax1.set_ylabel('True Positive Rate', font1)
    # title1 = 'Cross Validated ROC Curve'
    # ax1.set_title(title1, font1)
    ax1.legend(loc="lower right")
    figure1.savefig(folder + '5_fold_roc.png', dpi=300, bbox_inches = 'tight')

    ########PR_figure
    figure2, ax2 = plt.subplots(figsize=figsize)
    ax2.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 

    precision, recall, _ = precision_recall_curve(allylable, allprobas_)
    ax2.plot(recall, precision, color='b',
            label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(allylable, allprobas_)),
            lw=2, alpha=.8)

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall', font1)
    ax2.set_ylabel('Precision', font1)
    # title2 = 'Cross Validated PR Curve'
    # ax2.set_title(title2, font1)
    ax2.legend(loc="lower left")
    figure2.savefig(folder + '5_fold_pr.png', dpi=300, bbox_inches = 'tight')

if __name__ == '__main__':
    main_model_prediction()
