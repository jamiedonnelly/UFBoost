import sys
import os
import numpy as np 
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import os

fdir = "..."
FIGHTERS = [i.replace('.json','').replace("_"," ") for i in os.listdir(fdir) if ".DS" not in i]

# Checking a fighters weightclass 

weight_dict = {'Flyweight':1,'Bantamweight':2,'Featherweight':3,'Lightweight':4,\
                 'Welterweight':5,'Middleweight':6,'Light Heavyweight':7,'Heavyweight':8}

def calculate_weight_stats(data: pd.DataFrame):
    scores = {}
    base_dict = {'Flyweight':[0],'Bantamweight':[0],'Featherweight':[0],'Lightweight':[0],\
                 'Welterweight':[0],'Middleweight':[0],'Light Heavyweight':[0],'Heavyweight':[0]}
    for key in base_dict.keys():
        data[f"A_n_{key}"]=0
        data[f"B_n_{key}"]=0
    for fighter in FIGHTERS:
        stats = {'Flyweight':[0],'Bantamweight':[0],'Featherweight':[0],'Lightweight':[0],\
                 'Welterweight':[0],'Middleweight':[0],'Light Heavyweight':[0],'Heavyweight':[0]}
        df = data[(data['A_fighter']==fighter) | (data['B_fighter']==fighter)].sort_values(by=["Date"],ascending=True)
        if len(df)==0:
            scores[fighter] = {key:stats[key][-1] for key in stats.keys()}
        else:
            for i in range(df.shape[0]):
                if df.loc[df.index[i],'A_fighter']==fighter:
                    for key in base_dict.keys():
                        data.loc[df.index[i], f'A_n_{key}'] = stats[key][-1]
                elif df.loc[df.index[i],'B_fighter']==fighter:
                    for key in base_dict.keys():
                        data.loc[df.index[i], f'B_n_{key}'] = stats[key][-1]
                else:
                    continue
                division = df.loc[df.index[i],'division']
                if ((df.loc[df.index[i], 'A_fighter']==fighter) & (df.loc[df.index[i],'A_win']==True)) | ((df.loc[df.index[i], 'B_fighter']==fighter) & (df.loc[df.index[i],'B_win']==True)):
                    stats[division].append(stats[division][-1]+1)
                else:
                    stats[division].append(stats[division][-1]+1)
            scores[fighter] = {key:stats[key][-1] for key in stats.keys()}
    return scores

def prepare_arrays_base(stats, data: pd.DataFrame, train: bool = False):
    X, y = [], []
    for ix in data.index:
        # Ignore draws
        if (data.loc[ix,'A_win']=='False') & (data.loc[ix,'B_win']=='False'):
            continue
        
        if (data.loc[ix,'A_fighter'] not in FIGHTERS) or (data.loc[ix,'B_fighter'] not in FIGHTERS):
            continue
        
        A_fighter, B_fighter = data.loc[ix,'A_fighter'], data.loc[ix,'B_fighter']

        division = data.loc[ix,'division']   

        if train==True:
            current_division_stats_A, current_division_stats_B = np.array([data.loc[ix,f"A_n_{division}"]]), np.array([data.loc[ix,f"B_n_{division}"]])
        else:
            current_division_stats_A, current_division_stats_B = np.array([stats[A_fighter][division]]), np.array([stats[B_fighter][division]])
        
        if (weight_dict[division] - 1) in weight_dict.values():
            lower_division = list(weight_dict.keys())[list(weight_dict.values()).index(weight_dict[division] - 1)]
            if train == True:
                lower_A, lower_B = np.array([data.loc[ix, f"A_n_{lower_division}"]]), np.array([data.loc[ix, f"B_n_{lower_division}"]])
            else:
                lower_A, lower_B = np.array([stats[A_fighter][f"{lower_division}"]]), np.array([stats[B_fighter][f"{lower_division}"]])
        else:
            lower_A, lower_B = np.array([0.0]), np.array([0.0])

        if (weight_dict[division] + 1) in weight_dict.values():
            upper_division = list(weight_dict.keys())[list(weight_dict.values()).index(weight_dict[division] + 1)]
            if train == True:
                upper_A, upper_B = np.array([data.loc[ix, f"A_n_{upper_division}"]]), np.array([data.loc[ix, f"B_n_{upper_division}"]])
            else:
                upper_A, upper_B = np.array([stats[A_fighter][f"{upper_division}"]]), np.array([stats[B_fighter][f"{upper_division}"]])
        else:
            upper_A, upper_B = np.array([0.0]), np.array([0.0])

        X.append(np.concatenate([
            np.array([weight_dict[division]]),\
            current_division_stats_A,\
            current_division_stats_B,\
            lower_A,\
            lower_B,\
            upper_A,\
            upper_B
        ]))
        y.append(np.array([data.loc[ix,'A_win']], dtype=int))
    
    return np.array(X), np.array(y)

def prepare_features(train_df, test_df):
    # calculate stats from training data
    scores = calculate_weight_stats(train_df)
    # training arrays
    Xtr, ytr = prepare_arrays_base(scores, train_df, train=True)
    
    Xts, yts = prepare_arrays_base(scores, test_df)
    
    return (Xtr, ytr), (Xts, yts)


if __name__=="__main__":
    # Load arguments
    train_max_date = sys.argv[1]

    # Load data 
    data = pd.read_csv("/Users/jamie/Documents/MMA/data/train_fights.csv").sort_values(by=['Date'],ascending=True)

    # Partition data 
    train_data, test_data = data[(data['Date']<=train_max_date)], data[(data['Date']>train_max_date)]
    test_dates = test_data['Date'].unique()

    # Train model 
    td = test_dates[0]
    train_df = data[data['Date']<td]
    test_df = data[data['Date']==td]

    (Xtr, ytr), (Xts, yts) = prepare_features_weight(train_df, test_df)

    # transformation 
    mu, std = np.mean(Xtr, axis=0), np.std(Xtr, axis=0)
    Xtr = (Xtr-mu)

    # Input perturbation
    Xtr, ytr = random_shift(Xtr, ytr, scale=0.25, p=0.15)

    # Fit model 
    model = XGBClassifier(max_depth=12, n_estimators=25, objective='binary:logistic', gamma=1.0, reg_lambda=1.0)
    model.fit(Xtr, ytr.flatten())

    # Save model 
    model_path = '/Users/jamie/Documents/MMA/models/weight.json'
    model.save_model(model_path)




