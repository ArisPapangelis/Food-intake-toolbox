# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:16:15 2021

@author: Aris Papangelis

Extract in-meal indicators from every participant and save them to a .csv

"""

from extract_cfi import extract_cfi
import os
import pandas as pd


cols = ['Participant', 'a', 'b', 'Total food intake','Average food intake rate', 
        'Average bite size', 'Bite size std', 'Bites per minute']

#Dataset Katerinas
katerina_indicators = pd.concat([pd.DataFrame([(str(i),) + extract_cfi("data_katerinas", str(i),10,True,1)], columns = cols) 
                                 for i in range(1,29)], ignore_index=True)

katerina_indicators.to_csv("csv/katerina_indicators.csv", index=False, sep = ';' )



#Dataset clemson cafeteria    
files = os.listdir("clemson_data")
participants = [f.split('.')[0] for f in files]
participants = participants[:-1]

clemson_indicators = pd.concat([pd.DataFrame([(p,) + extract_cfi("clemson_data", p, 15,True,1)], columns = cols) 
                                 for p in participants], ignore_index=True)

clemson_indicators.to_csv("csv/clemson_indicators.csv", index=False, sep = ';' )

