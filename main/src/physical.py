import pandas as pd 
import numpy as np 
from imblearn.over_sampling import RandomOverSampler
import os
import json
from datetime import datetime

fdir = "..."
FIGHTERS = [i.replace('.json','').replace("_"," ") for i in os.listdir(fdir) if ".DS" not in i]

def calculate_age(birth_date_str, date_str):
    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    delta = date - birth_date
    
    age = delta.days / 365.25
    
    return age

def calculate_age_delta(date, A_age, B_age):
    # calculate each fighters age at time of fight
    A_dt = datetime.strptime(A_age, '%Y-%m-%d')
    B_dt = datetime.strptime(B_age, '%Y-%m-%d')
    # Delta 
    difference = A_dt - B_dt
    return round(difference.days/365.25,3)

def prepare_arrays_base(data: pd.DataFrame, train: bool = False):
    X, y = [], []
    for ix in data.index:
        # Ignore draws
        if (data.loc[ix,'A_win']=='False') & (data.loc[ix,'B_win']=='False'):
            continue
        
        if (data.loc[ix,'A_fighter'] not in FIGHTERS) or (data.loc[ix,'B_fighter'] not in FIGHTERS):
            continue
        
        A_fighter, B_fighter = data.loc[ix,'A_fighter'], data.loc[ix,'B_fighter']
        
        with open(os.path.join(fdir,A_fighter.replace(" ","_")+'.json'),'r') as f:
            A_stats = json.load(f)
        with open(os.path.join(fdir,B_fighter.replace(" ","_")+'.json'),'r') as f:
            B_stats = json.load(f)
        
        date = data.loc[ix,'Date']

        A_age, B_age = calculate_age(A_stats['dob'], date), calculate_age(B_stats['dob'], date)
                
        # height delta 
        height_delta = A_stats['height'] - B_stats['height']

        # reach delta 
        reach_delta = A_stats['reach'] - B_stats['reach']

        # stance delta 
        if A_stats['stance']==B_stats['stance']:
            stance_delta = 0
        else:
            stance_delta = 1
        
        X.append(np.array([A_age, B_age, height_delta, reach_delta, stance_delta]))
        y.append(np.array([data.loc[ix,'A_win']],dtype=int))
    
    return np.array(X), np.array(y)

def prepare_features(train_df, test_df):
    # training arrays
    Xtr, ytr = prepare_arrays_base(train_df)
    
    Xts, yts = prepare_arrays_base(test_df)

    return (Xtr, ytr), (Xts, yts)