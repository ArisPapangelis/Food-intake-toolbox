# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:08:49 2021

@author: Aris Papangelis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#Function used to detect food additions through greedy quadratic fitting (not used currently)
def detect_FA(time, cfi, ranges, i, second_to_first_difference):
    
    #Current meal trend
    current_coeff = np.polyfit(time[:ranges[i-1,1]], cfi[:ranges[i-1,1]], 2)
    current_curve = np.polyval(current_coeff, time[:ranges[i,1]])
    current_curve = current_curve - current_curve[0]
    
    #Future meal trend if food addition is denied
    NFA_coeff = np.polyfit(time[:ranges[i,1]], cfi[:ranges[i,1]], 2)
    NFA_curve = np.polyval(NFA_coeff, time[:ranges[i,1]])
    NFA_curve = NFA_curve - NFA_curve[0]
    
    #Future meal trend if food addition is accepted
    FA = np.copy(cfi)
    FA[:ranges[i-1,1]+1] += second_to_first_difference
    FA_coeff = np.polyfit(time[:ranges[i,1]], FA[:ranges[i,1]], 2)
    FA_curve = np.polyval(FA_coeff, time[:ranges[i,1]])
    FA_curve = FA_curve - FA_curve[0]
    
    #Plot the curves
    plt.plot(time[:ranges[i,1]], current_curve, label = 'Current curve trend')
    plt.plot(time[:ranges[i,1]], NFA_curve, label = 'Denied food addition curve trend')
    plt.plot(time[:ranges[i,1]], FA_curve, label = 'Accepted food addition curve trend')
    
    #Decide if there is an actual food addition based on the mean squared error of the curves
    mse_FA = np.sum(np.square(current_curve-FA_curve)) / len(current_curve)
    mse_NFA = np.sum(np.square(current_curve-NFA_curve)) / len(current_curve)
    if mse_FA < mse_NFA:
        return True
    else:
        return False


#Compare ground truth bites to detected bites
def bite_detection_metrics(file, downsampling_factor, bite_indices):
    
    #Read ground truth bite timestamps
    gt_bites = pd.read_csv("clemson_gt_bites/{}.txt".format(file), delimiter = ':', header=None, names=['Bite Index', 'Container'], skiprows=1)
    gt_bites['Bite Index'] = (gt_bites['Bite Index'] / downsampling_factor).astype(int)
    bites_gt = gt_bites['Bite Index'].to_numpy()
    
    #Match detected bites to ground truth bites
    gt_i, i = 0, 0
    tp=0
    while i < len(bite_indices) and gt_i < len(bites_gt):
        bite_difference = bite_indices[i] - bites_gt[gt_i]
        if abs(bite_difference) < 50:
            tp +=1
            i+=1
            gt_i+=1
        else:
            if bite_difference < 0:
                i+=1
            else:
                gt_i+=1
    
    #Calculate bite detection metrics
    fn = len(bites_gt) - tp
    fp = len(bite_indices) - tp
    precision = 0
    recall = 0
    f1 = 0
    if tp + fp != 0:
        precision = tp / (tp+fp)
    if tp + fn != 0:
        recall = tp / (tp+fn)
    if precision + recall != 0: 
        f1 = 2 * precision * recall / (precision + recall)  
    
    #Write calculated metrics for the current file to csv
    cols = ['File', 'TP', 'FP', 'FN', 'Precision','Recall', 'F1']        
    bite_metrics = pd.DataFrame([(file, tp, fp, fn, precision, recall, f1)], columns = cols)
    
    if not os.path.exists('csv/bite_metrics.csv'):    
        bite_metrics.to_csv('csv/bite_metrics.csv',  index=False, sep = ';')
    else:
        bite_metrics.to_csv('csv/bite_metrics.csv', mode='a',  index=False, sep = ';', header=False)
        
    return bites_gt
    