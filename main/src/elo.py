import os
import numpy as np 
import pandas as pd
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
import json

fdir = "..."
FIGHTERS = [i.replace('.json','').replace("_"," ") for i in os.listdir(fdir) if ".DS" not in i]

def result_score(fighter, fight):
    if ((fight['A_fighter']==fighter) & (fight['A_win']==True)) or ((fight['B_fighter']==fighter) & (fight['B_win']==True)):
        if fight['method'] == 'ko/tko':
            return 9
        else:
            return 3
    elif (fight['A_win']=='False') & (fight['B_win']=='False'):
        return 0 
    else:
        if fight['method'] == 'ko/tko':
            return -9
        else:
            return -3
        
def title_result(score, fighter, fight, fight_id, data):
    added_score = 0
    
    if (fight['title_bout']==True) & (score>=3):
        added_score += 3
        
    if (fight['title_bout']==True) & (score<0):
        current_date = datetime.strptime(data.loc[fight_id,'Date'],'%Y-%m-%d')
        for ix in range(fight_id-1,-1,-1):
            row = data.iloc[ix,:]
            old_date = datetime.strptime(row['Date'],'%Y-%m-%d')
            delta = current_date - old_date
            if delta.days > 700:
                break
            else:
                if ((row['A_fighter']==fighter) or (row['B_fighter']==fighter)) & (row['title_bout']==True):
                    if result_score(fighter, row)>=3:
                        #title loss
                        added_score -= 2
    return added_score

def current_record(fighter, fight_id, data):
    record = []
    current_date = datetime.strptime(data.loc[fight_id,'Date'],'%Y-%m-%d')
    for ix in range(fight_id-1,-1,-1):
        row = data.iloc[ix,:]
        old_date = datetime.strptime(row['Date'],'%Y-%m-%d')
        delta = current_date - old_date
        if delta.days > 700:
            break
        else:
            if (row['A_fighter']==fighter) or (row['B_fighter']==fighter):
                score = result_score(fighter, row)
                if score>0:
                    record.append(1)
                elif score:
                    record.append(-1)
    return record
                  
def streak(fighter, fight_id, data):
    record = current_record(fighter, fight_id, data)
    if len(record)==0:
        return 1
    elif record[0]==-1:
        count = 1
        for ix, i in enumerate(record[1:]):
            if i == -1:
                count += 1
            else:
                break
        return 1.1**count
    return 1
    
def difficulty_multiplier(scores, fighter, opp, res):
    if (scores[opp]>scores[fighter]) & (res<0):
        return (scores[fighter]/scores[opp])
    if (scores[opp]>scores[fighter]) & (res>0):
        return (scores[opp]/scores[fighter])
    if (scores[fighter]>scores[opp]) & (res<0):
        return (scores[fighter]/scores[opp])
    if (scores[fighter]>scores[opp]) & (res>0):
        return (scores[opp]/scores[fighter])
    else:
        return 1
    
def fight_scores(fight, fighter):
    A, B = fight['A_fighter'], fight['B_fighter']
    net_kd = fight['A_KD'] - fight['B_KD']
    net_td = fight['A_TD'] - fight['B_TD']
    net_head_str = fight['A_Head'] - fight['B_Head']
    net_body_str = fight['A_Body'] - fight['B_Body']
    net_leg_str = fight['A_Leg'] - fight['B_Leg']
    net_ctrl = fight['A_CTRL'] - fight['B_CTRL']
    score = (0.25*net_kd) + (0.1*net_td) + (0.02*net_head_str) + (0.01*net_body_str) + (0.005*net_leg_str) +(0.01*net_ctrl)
    if fighter==A:
        return score
    else:
        return -1*score
    
def calculate_score(scores,fighter, opp, fight, idx, data):
    outcome = result_score(fighter, fight)
    ts = title_result(outcome, fighter, fight, idx, data)
    fs = fight_scores(fight, fighter)
    agg = outcome + ts + fs
    diff_mul = difficulty_multiplier(scores,fighter, opp, agg)
    return agg*(diff_mul)
        
def calculate_stats(data: pd.DataFrame):
    scores = {}
    for f in FIGHTERS:
        scores[f] = 100
    for ix in range(data.shape[0]):
        fight = data.iloc[ix,:]
        if (fight['A_fighter'] in scores.keys()) & (fight['B_fighter'] in scores.keys()):   
            data.loc[fight.name,'A_elo'] = scores[fight['A_fighter']]
            data.loc[fight.name,'B_elo'] = scores[fight['B_fighter']]
            scores[fight['A_fighter']] += calculate_score(scores,fight['A_fighter'],fight['B_fighter'],fight,ix,data)
            scores[fight['B_fighter']] += calculate_score(scores,fight['B_fighter'],fight['A_fighter'],fight,ix,data)
        else:
            continue
    return scores

def calculate_age(birth_date_str, date_str):
    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    delta = date - birth_date
    
    age = delta.days / 365.25
    
    return age
        
def decay_scores_df(scores, data, date):
    ann_decay = 0.90
    current_date = datetime.strptime(date,'%Y-%m-%d')
    for fighter in FIGHTERS:
        df = data[(data['A_fighter']==fighter)|(data['B_fighter']==fighter)].sort_values(by=['Date'],ascending=True)
        if len(df)==0:
            continue
        else:
            for i in range(1,len(df)):
                dt1, dt2 = datetime.strptime(df.loc[df.index[i-1],'Date'],'%Y-%m-%d'),\
                datetime.strptime(df.loc[df.index[i],'Date'],'%Y-%m-%d')
                delta = dt2-dt1
                num_years = delta.days / 365.25 
                if df.loc[df.index[i],'A_fighter'] == fighter:
                    data.loc[df.index[i],'A_elo']*=(ann_decay**num_years)
                else:
                    data.loc[df.index[i],'B_elo']*=(ann_decay**num_years)
            last_fight = datetime.strptime(df['Date'].max(),'%Y-%m-%d')
            delta = current_date - last_fight 
            num_years = delta.days / 365.25
            scores[fighter] = scores[fighter]*(ann_decay**num_years)  

def age_weighted_score_df(fdir, scores, data, date):
    for fighter in FIGHTERS:
        fpath = os.path.join(fdir, fighter.replace(" ","_")+'.json')
        with open(fpath, 'r') as f:
            stats = json.load(f)
        # modify historical values in dataframe
        df = data[(data['A_fighter']==fighter)|(data['B_fighter']==fighter)].sort_values(by=['Date'],ascending=True)
        if len(df)==0:
            pass
        else:
            for i in range(1,len(df)):
                age = calculate_age(stats['dob'],df.loc[df.index[i],'Date'])
                if age<33:
                    pass
                else:
                    if df.loc[df.index[i],'A_fighter']==fighter:
                        data.loc[df.index[i],'A_elo'] *= (0.95)**(age-33)
                    elif df.loc[df.index[i],'B_fighter']==fighter:
                        data.loc[df.index[i],'B_elo'] *= (0.95)**(age-33)
        # modify current scores in dict
        age = calculate_age(stats['dob'],date)
        if age<33:
            pass
        else:
            scores[fighter] = (0.95)**(age-33)*scores[fighter]

def avg_opponent_elo(data):
    avg_opp_elo = {}
    data['A_avg_opp_elo']=100
    data['B_avg_opp_elo']=100
    for f in FIGHTERS:
        avg_opp_elo[f] = 100
        df = data[(data['A_fighter']==f)|(data['B_fighter']==f)].sort_values(by=['Date'],ascending=True)
        if len(df)==0:
            continue
        else:
            for i in range(df.shape[0]):
                mu = [100]
                for j in range(i):
                    if df.loc[df.index[j],'A_fighter']==f:
                        mu.append(df.loc[df.index[j],'B_elo'])
                    elif df.loc[df.index[j],'B_fighter']==f:
                        mu.append(df.loc[df.index[j],'A_elo'])
                    else:
                        continue
                if data.loc[df.index[i],'A_fighter']==f:
                    data.loc[df.index[i],'A_avg_opp_elo'] = sum(mu)/len(mu)
                elif data.loc[df.index[i],'B_fighter']==f:
                    data.loc[df.index[i],'B_avg_opp_elo'] = sum(mu)/len(mu)
            avg_opp_elo[f] = sum(mu)/len(mu)
    return avg_opp_elo

def prepare_arrays_base(elo, avg_opp_elo, data: pd.DataFrame, train: bool = False):
    X, y = [], []
    for ix in data.index:
        # Ignore draws
        if (data.loc[ix,'A_win']=='False') & (data.loc[ix,'B_win']=='False'):
            continue
        
        if (data.loc[ix,'A_fighter'] not in FIGHTERS) or (data.loc[ix,'B_fighter'] not in FIGHTERS):
            continue
        
        A_fighter, B_fighter = data.loc[ix,'A_fighter'], data.loc[ix,'B_fighter']
        
        if train==True:
            X.append(np.array([data.loc[ix,'A_elo'], data.loc[ix,'A_avg_opp_elo'],\
                               data.loc[ix,'B_elo'], data.loc[ix,'B_avg_opp_elo']]))
        else:
            X.append(np.array([elo[A_fighter], avg_opp_elo[A_fighter],\
                               elo[B_fighter], avg_opp_elo[B_fighter]]))
        y.append(np.array([data.loc[ix,'A_win']], dtype=int))
    
    return np.array(X), np.array(y)

def prepare_features(train_df, test_df, date):
    elo = calculate_stats(train_df)
    decay_scores_df(elo, train_df, date)
    age_weighted_score_df(fdir, elo, train_df, date)
    avg_opp_elo = avg_opponent_elo(train_df)
    # training arrays
    Xtr, ytr = prepare_arrays_base(elo, avg_opp_elo, train_df, train=True)
    Xts, yts = prepare_arrays_base(elo, avg_opp_elo, test_df)
    return (Xtr, ytr), (Xts, yts)

