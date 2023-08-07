import numpy as np 
from imblearn.over_sampling import RandomOverSampler
import os

fdir = "..."
FIGHTERS = [i.replace('.json','').replace("_"," ") for i in os.listdir(fdir) if ".DS" not in i]

def prepare_arrays(stats, data, train=False):
    X, y = [], []
    for ix in data.index:
        # Ignore draws
        if (data.loc[ix,'A_win']=='False') & (data.loc[ix,'B_win']=='False'):
            continue
        
        if (data.loc[ix,'A_fighter'] not in FIGHTERS) or (data.loc[ix,'B_fighter'] not in FIGHTERS):
            continue
        
        A_fighter, B_fighter = data.loc[ix,'A_fighter'], data.loc[ix,'B_fighter']
        
        if train==True:
            X.append(np.array([data.loc[ix,'A_streak'], data.loc[ix,'B_streak']]))
        else:
            X.append(np.array([stats[A_fighter], stats[B_fighter]]))
        y.append(np.array([data.loc[ix,'A_win']], dtype=int))
    
    return np.array(X), np.array(y)

def prepare_features(train_df, test_df):
    streaks = {}
    train_df['A_streak']=0
    train_df['B_streak']=0
    for f in FIGHTERS:
        streak = [0]
        df = train_df[(train_df['A_fighter']==f)|(train_df['B_fighter']==f)].sort_values(by=['Date'],ascending=True)
        for i in range(df.shape[0]):
            if (df.loc[df.index[i],'A_fighter']==f) & (df.loc[df.index[i],'A_win']==True):
                if streak[-1]<=0:
                    # if current losing streak, reset streak to 1
                    streak.append(1)
                else:
                    # if not losing then add 1 to current streak
                    streak.append(streak[-1]+1)
            elif (df.loc[df.index[i],'B_fighter']==f) & (df.loc[df.index[i],'A_win']==False):
                if streak[-1]<=0:
                    # if current losing streak, reset streak to 1
                    streak.append(1)
                else:
                    # if not losing then add 1 to current streak
                    streak.append(streak[-1]+1)
            else:
                if streak[-1]<=0:
                    # if streak is already 0 or negative then decrement by 1
                    streak.append(streak[-1]-1)
                else:
                    # if streak was positive reset streak to 0
                    streak.append(0)
        for i in range(df.shape[0]):
            if train_df.loc[df.index[i],'A_fighter']==f:
                train_df.loc[df.index[i],'A_streak'] = streak[i]
            elif train_df.loc[df.index[i],'B_fighter']==f:
                train_df.loc[df.index[i],'B_streak'] = streak[i]
            else:
                continue
        streaks[f] = streak[-1]
    Xtr, ytr = prepare_arrays(streaks, train_df, train=True)
    Xts, yts = prepare_arrays(streaks, test_df)
    return (Xtr, ytr), (Xts, yts)

