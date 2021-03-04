# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 04:03:56 2020

@author: Aris Papangelis

Function that extracts the CFI curve from each meal, as well as the in-meal indicators

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper_functions import bite_detection_metrics



def fit_func(x, a, b):
  return a * x ** 2 +  b * x 


#https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges




def extract_cfi(folder, file, initial_sampling_rate, end_of_meal, stable_secs):

    #22 is food addition
    #23 is food mass bite
    #9,15,24,26 problematic
    
    filepath = folder + "/" + file + ".txt"
    time, cfi = np.loadtxt(filepath, delimiter = ':', skiprows=1, unpack = True)
    
    
    
    #Downsampling 10Hz to 5Hz
    downsampled_rate = 5
    downsampling_factor = int(initial_sampling_rate / downsampled_rate)
    time = time[::downsampling_factor]
    cfi = cfi[::downsampling_factor]
    
    cfi_raw = cfi.copy()
    
    plt.ioff()
    plt.figure(file, figsize=(19.2, 10.8))
    plt.xlabel('Time')
    plt.ylabel('Weight')
    plt.plot(time,cfi, label = "Initial data")
    
    
    
    #Find stable periods
    stability_threshold = 1
    diff = abs(np.diff(cfi))
    diff = np.where(diff <= stability_threshold, 0, diff)
    ranges = zero_runs(diff)
    
    #Keep only the stable periods that last more than stable_secs seconds
    stable_period_length = stable_secs * downsampled_rate
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
    

    #Compare stable periods to eliminate artifacts or identify food additions
    food_mass_bite_threshold = 75
    food_addition_threshold = 60
    food_mass_bite_count = 0
    for i in range(1,len(ranges)):
                
        #Food addition
        second_to_first_difference = cfi[ranges[i,0]+1]-cfi[ranges[i-1,0]+1]
        if second_to_first_difference > food_addition_threshold:
            deltaDiff = delta[ranges[i,0]+1] - delta[ranges[i-1,0]+1]
            #deltaDiff>=0 and 
            if deltaDiff>=0 and delta[ranges[i,0]+1] > 0 :
                for j in range(ranges[i-1,1]+1):
                    cfi[j] = cfi[j] + second_to_first_difference
            else:
                cfi[ranges[i,0]+1:ranges[i,1]+1] = cfi[ranges[i-1,0]+1]
        
        
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
            
    
    
    #"""
    #Start and end of meal
    cfi = cfi[ranges[0,0]:ranges[len(ranges)-1,1]]
    time = time[ranges[0,0]:ranges[len(ranges)-1,1]]

    if end_of_meal == True:
        print(file)
        indices = np.where(np.diff(cfi)!=0)[0]
        if len(indices) != 0:
            startIndex = indices[0]
            endIndex = indices[-1]
            start = startIndex - 20 if startIndex - 20 >= 0 else startIndex - 10
            end = endIndex + 20 if endIndex + 20 < len(cfi) else endIndex + 10
            cfi = cfi[start:end]
            time = time[start:end]
    
    
    #Experimentation with plate weight for training mode
    plate_weight = 559 #1
    #plate_weight = 652.1 #2
    #"""
    if plate_weight > 5:
        end_weight = cfi[0] - plate_weight
        reference_coeff = [-0.0005, 1, -end_weight]
        roots = np.roots(reference_coeff)
        if np.isreal(roots[0]) and np.isreal(roots[1]): 
            candidate_times = np.real(min(roots))
        else:
            candidate_times = np.real(np.sqrt(roots[0]*roots[1]))
        reference_time = np.arange(0, candidate_times, 1/5)
        reference_curve = reference_coeff[0] * reference_time ** 2 + reference_coeff[1] * reference_time
        plt.plot(reference_time, reference_curve, label= "Reference curve with roots")
    #"""  
    #"""
    if plate_weight > 5:
        end_weight = cfi[0] - plate_weight
        #reference_coeff = [-0.0005, 1, -end_weight]
        time_to_finish = end_weight / 0.8
        reference_time = np.arange(0, time_to_finish, 1/5)
        reference_weight = np.linspace(0, end_weight, num = len(reference_time))
        #reference_time = np.array([0, time_to_finish])
        #reference_weight = np.array([0, end_weight])
        reference_coeff = curve_fit(fit_func, reference_time, reference_weight,
                                    bounds = ([-1, 0], [-0.0005, 2]))[0]
        reference_curve = reference_coeff[0] * reference_time ** 2 + reference_coeff[1] * reference_time
        plt.plot(reference_time, reference_curve, label= "Reference curve with curve fit")
    #"""
    
    
    #"""
    index_offset = time[0] * downsampled_rate
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
    coefficients = curve_fit(fit_func,time,cfi)[0]
    #coefficients[-1] = 0
    curve = coefficients[0] * time ** 2 + coefficients[1] * time
    plt.plot(time,curve, label = "Quadratic curve")
   
    
    #a, b, total food intake, average food intake rate, average bite size and standard deviation, bites per minute
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
    
    #Recursively call extract cfi with a modified stable period length if 1 sec doesn't work,
    #to eliminate some long lasting stable spikes
    if len(bites) == 0 and stable_secs <= 3:
        plt.close(file)
        return extract_cfi(folder, file, initial_sampling_rate, end_of_meal, stable_secs + 1)
    
    #Ground truth data comparison for the clemson cafeteria dataset
    if folder == "clemson_data":
        bites_gt = bite_detection_metrics(file, downsampling_factor, bite_indices + int(index_offset))
        plt.scatter(bites_gt / downsampled_rate, cfi_raw[bites_gt], label = 'Ground Truth bites', c = 'tab:pink')
    

    
    #Plot extracted cfi curve
    plt.plot(time,cfi, label = "Extracted CFI")
    plt.scatter((bite_indices + int(index_offset)) / downsampled_rate, cfi_raw[bite_indices + int(index_offset)], label = 'Detected bites', c = 'tab:orange')
    plt.legend()
    plt.savefig(folder + "/pics/" + file +".png")
    plt.show()
    #plt.close(file)


    #results = np.array([a, b, total_food_intake, average_food_intake_rate, average_bite_size, bite_size_STD, bite_frequency])
    return a, b, total_food_intake, average_food_intake_rate, average_bite_size, bite_size_STD, bite_frequency




if __name__=="__main__":
    #extract_cfi("clemson_data", 'p011_c1', 15, True, 1)
    extract_cfi("data_katerinas", str(1), 10, True, 1)
