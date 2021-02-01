# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 04:03:56 2020

@author: Aris Papangelis

Experimenting on the CFI extraction algorithm, ignore this file for final implementation

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os

def fit_func(x, a, b):
  return a * x ** 2 +  b * x 


#https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    #import inspect
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    #local = inspect.currentframe().f_locals
    return ranges




#22 is food addition
#7,13,26 also food additions
#23 is food mass bite
#9,15,24,26 problematic

file = str(23)
#file = "p005_c1"
filepath = "data_katerinas/" + file + ".txt"
#filepath = "clemson_data/" + file + ".txt"
time, cfi = np.loadtxt(filepath, delimiter = ':', skiprows=1, unpack = True)


#Downsampling 10Hz to 5Hz
initial_sampling_rate = 15
downsampled_rate = 5
downsampling_factor = int(initial_sampling_rate / downsampled_rate)
time = time[::downsampling_factor]
cfi = cfi[::downsampling_factor]


#cfi_init = cfi.copy()
plt.ioff()
fig = plt.figure(file, figsize=(19.2, 10.8))
plt.xlabel('Time')
plt.ylabel('Weight')
plt.plot(time,cfi, label = "Initial data")

plt.show()




#Find stable periods
stability_threshold = 1
diff = abs(np.diff(cfi))
diff = np.where(diff <= stability_threshold, 0, diff)
ranges = zero_runs(diff)

#Keep only the stable periods that last more than 1 seconds
stable_period_length = 5
diffranges = np.diff(ranges,axis=1)
ranges = ranges[np.where(diffranges>=stable_period_length)[0]]


#Set the same sample value for each stable period, to eliminate jitter
cfi[ranges[0,0]:ranges[0,1]+1] = np.median(cfi[ranges[0,0]:ranges[0,1]+1]) 
for i in range(1,len(ranges)):
        cfi[ranges[i,0]+1:ranges[i,1]+1] = np.median(cfi[ranges[i,0]+1:ranges[i,1]+1]) 

        
#Delta coefficients
if time[-1] < 60:
    secs = int(time[-1])
else:
    secs = 60
    
D = int(secs / 2 * downsampled_rate)
taps = np.arange(-D, D + 1)
denominator = np.sum(np.square(taps))
fir = taps / denominator

delta = -100 * np.convolve(fir,cfi, mode = 'valid')

padding = int((len(cfi)-len(delta)) / 2)
delta = np.pad(delta, (padding,padding), mode = 'edge')

plt.plot(time,delta, label = "Delta")



#Compare stable periods to eliminate artifacts or identify food additions and food mass bites
food_mass_bite_threshold = 75
food_addition_threshold = 60
food_mass_bite_count = 0
for i in range(1,len(ranges)):
    
    #Initial crude attempt to detect food additions
    """
    if i<len(ranges)-1 and second_to_first_difference > food_addition_threshold:
        third_to_first_difference = cfi[ranges[i+1,0]+1]-cfi[ranges[i-1,0]+1]
        if third_to_first_difference > food_addition_threshold / 2:
            for j in range(ranges[i-1,1]+1):
                cfi[j] = cfi[j] + second_to_first_difference
        else:
            cfi[ranges[i,0]+1:ranges[i,1]+1] = cfi[ranges[i-1,0]+1]
    """
    
    #Food addition
    second_to_first_difference = cfi[ranges[i,0]+1]-cfi[ranges[i-1,0]+1]
    if second_to_first_difference > food_addition_threshold:
        #Attempt to detect food additions with GQF (greedy quadratic fitting).
        """
        current_coeff = np.polyfit(time[:ranges[i-1,1]], cfi[:ranges[i-1,1]], 2)
        current_curve = np.polyval(current_coeff, time[:ranges[i,1]])
                                                       
        current_curve = current_curve - current_curve[0]
        
        NFA_coeff = np.polyfit(time[:ranges[i,1]], cfi[:ranges[i,1]], 2)
        NFA_curve = np.polyval(NFA_coeff, time[:ranges[i,1]])
        NFA_curve = NFA_curve - NFA_curve[0]
        
        FA = np.copy(cfi)
        FA[:ranges[i-1,1]+1] += second_to_first_difference
        
        FA_coeff = np.polyfit(time[:ranges[i,1]], FA[:ranges[i,1]], 2)
        FA_curve = np.polyval(FA_coeff, time[:ranges[i,1]])
        FA_curve = FA_curve - FA_curve[0]
        
        plt.plot(time[:ranges[i,1]], current_curve, label = 'Current curve trend')
        plt.plot(time[:ranges[i,1]], NFA_curve, label = 'Denied food addition curve trend')
        plt.plot(time[:ranges[i,1]], FA_curve, label = 'Accepted food addition curve trend')
        MSE_FA = np.sum(np.square(current_curve-FA_curve)) / len(current_curve)
        MSE_NFA = np.sum(np.square(current_curve-NFA_curve)) / len(current_curve)

        plt.legend()
        plt.show()
        print('stop')
        if MSE_FA < MSE_NFA:
            for j in range(ranges[i-1,1]+1):
                cfi[j] = cfi[j] + second_to_first_difference
        else:
            cfi[ranges[i,0]+1:ranges[i,1]+1] = cfi[ranges[i-1,0]+1]
            
        """

        #"""
        deltaDiff = delta[ranges[i,0]+1] - delta[ranges[i-1,0]+1]
        if deltaDiff>=0 and delta[ranges[i,0]+1] > 0:
            for j in range(ranges[i-1,1]+1):
                cfi[j] = cfi[j] + second_to_first_difference
        else:
            cfi[ranges[i,0]+1:ranges[i,1]+1] = cfi[ranges[i-1,0]+1]
        #"""
    
    
    #Large food mass bite
    elif second_to_first_difference < -food_mass_bite_threshold:
        #if i<len(ranges)-1 and cfi[ranges[i+1,0]+1]-cfi[ranges[i,0]+1]>10:
        cfi[ranges[i,0]+1:ranges[i,1]+1] = cfi[ranges[i-1,0]+1]
        food_mass_bite_count += 1
    

    #Artifacts
    elif second_to_first_difference>0:
        
        #Final food mass bites of the meal
        if i>1 and cfi[ranges[i,0]+1] - cfi[ranges[i-2,0]+1] < 0 and food_mass_bite_count > 5:
             cfi[ranges[i-1,0]+1:ranges[i-1,1]+1] = cfi[ranges[i-2,0]+1]
        
        #Utensilising artifact     
        else:     
            cfi[ranges[i,0]+1:ranges[i,1]+1] = cfi[ranges[i-1,0]+1]
            
        

#Find stable samples
#https://stackoverflow.com/questions/6036837/a-numpy-arange-style-function-with-array-inputs
stableSamples = np.concatenate([np.arange(x, y) for x, y in zip(ranges[:,0]+1, ranges[:,1]+1)])
stableSamples = np.concatenate((np.array([stableSamples[0]-1]), stableSamples))


#Set unstable samples equal to previous stable
for i in range(1,len(cfi)):
    if i not in stableSamples:
        cfi[i]=cfi[i-1]
        


"""
#Start of meal

cfi = cfi[ranges[0,1]-1:]
time = time[ranges[0,1]-1:]

cfi = cfi[ranges[0,0]:]
time = time[ranges[0,0]:]
startIndex = np.where(np.diff(cfi)!=0)[0][0]
cfi = cfi[startIndex-2:]
time = time[startIndex-2:]


#End of meal
cfi = cfi[:ranges[len(ranges)-1,0]+3]
time = time[:ranges[len(ranges)-1,0]+3]
"""



time = time - time[0]

#Remove plate weight and invert CFI curve
cfi = cfi-cfi[-1]
cfi = abs(cfi-cfi[0])

#Join consecutive stable periods of the same weight (eliminate bites below stability threshold)
bites = np.diff(cfi)
for i in np.where((bites!=0) & (bites <= stability_threshold))[0]:
    j = i+1
    weight = cfi[j]
    while (j<len(cfi) and cfi[j] == weight):
        cfi[j] = cfi[i]
        j+=1 


#Fit the extracted CFI curve to a second degree polynomial
#coefficients = np.polyfit(time,cfi,2)
coefficients = curve_fit(fit_func,time,cfi)[0]
#coefficients[-1] = 0
#curve = np.polyval(coefficients, time)
curve = coefficients[0] * time ** 2 + coefficients[1] * time
plt.plot(time,curve, label = "Quadratic curve")



#a, b, total food intake, average food intake rate
a = coefficients[0]
b = coefficients[1]
meal_duration = time[-1]
total_food_intake = cfi[-1]
average_food_intake_rate = total_food_intake / meal_duration


        
bites = np.diff(cfi)       
bite_indices = np.where(bites!=0)[0]
bites = bites[bite_indices]
average_bite_size = np.mean(bites)
bite_size_STD = np.std(bites)
bite_frequency = 60 * len(bites) / meal_duration

"""
#Compare ground truth bites to detected bites
gt_bites = pd.read_csv("clemson_gt_bites/{}.txt".format(file), delimiter = ':', header=None, names=['Bite Index', 'Container'], skiprows=1)
gt_bites['Bite Index'] = (gt_bites['Bite Index'] / downsampling_factor).astype(int)
bites_gt = gt_bites['Bite Index'].to_numpy()

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

fn = len(bites_gt) - tp
fp = len(bite_indices) - tp
precision = tp / (tp+fp)
recall = tp / (tp+fn)
f1 = 2 * precision * recall / (precision + recall)    


cols = ['File', 'TP', 'FP', 'FN', 'Precision','Recall', 'F1']        
bite_metrics = pd.DataFrame([(file, tp, fp, fn, precision, recall, f1)], columns = cols)

if not os.path.exists('csv/bite_metrics.csv'):    
    bite_metrics.to_csv('csv/bite_metrics.csv',  index=False, sep = ';')
else:
    bite_metrics.to_csv('csv/bite_metrics.csv', mode='a',  index=False, sep = ';', header=False)
"""

#Plot extracted cfi curve
plt.plot(time,cfi, label = "Extracted CFI")
plt.savefig("pics/"+str(file)+".png")
plt.legend()
plt.show()

#plt.cla()
#plt.gcf().show()