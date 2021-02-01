# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:57:02 2021

@author: Aris Papangelis

Script to bring the dataset to the correct form for further analysis

"""

import os
import numpy as np
import pandas as pd



participants = os.listdir("raw_clemson_data")

cols_to_use = ["Participant","Gender","Age","BMI","Body_Fat_%","Waist_(inches)","Hips_(inches)"]
demographics = pd.read_csv("demographics.csv", usecols = cols_to_use, decimal = ',', delimiter = ';')[cols_to_use]
demographics['Gender'].replace({'Female': 'F', 'Male': 'M'}, inplace = True)
demographics = demographics[demographics["Participant"].isin(participants)]
demographics = demographics[demographics["BMI"].notna()]
demographics.to_csv("csv/clemson_demographics.csv", index=False, sep = ';' )



for p in participants:
    courses = os.listdir("raw_clemson_data/{}".format(p))
    for c in courses:
        #Save meal weight data for each meal
        meal = os.listdir("raw_clemson_data/{}/{}".format(p,c))
        weight = np.loadtxt("raw_clemson_data/{}/{}/{}".format(p,c,meal[0]), usecols=-1, unpack = True)
        zero_glitches = np.where(weight==0)[0]
        for i in zero_glitches:
            weight[i] = weight[i-1]
                   
        #time = np.linspace(0, len(weight) / 15, num = len(weight))
        time = np.arange(0, len(weight) / 15, 1/15)
        time = time[:len(weight)]
  
        np.savetxt("clemson_data/{}_{}.txt".format(p,c), np.column_stack((time,weight)), fmt = "%f:%f", delimiter = ':',
                   header = "{} samples in {} seconds".format(len(weight), time[-1]))
        
        
        #Save ground truth bite data for each meal
        meal = os.listdir("raw_clemson_gt_bites/{}/{}".format(p,c))
        bite_index, container = np.genfromtxt("raw_clemson_gt_bites/{}/{}/{}".format(p,c,meal[0]), dtype=np.str, usecols=(1,-2), unpack = True)
        np.savetxt("clemson_gt_bites/{}_{}.txt".format(p,c), np.column_stack((bite_index,container)), fmt = "%s:%s", delimiter = ':',
                   header = "Ground truth bites for meal {}_{}".format(p,c))
        
