import numpy as np 
from sklearn.metrics import accuracy_score;
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def validate(classifier, Xts, yts, params = False) -> None:
    # Check for scaling params
    if not params:
        pass
    elif ('mean' in params.keys()) & ('std' in params.keys()):
        Xts = (Xts - params['mean'])/params['std']
    elif 'mean' in params.keys():
         Xts = Xts - params['mean']
    # predictions
    predictions = classifier.predict(Xts)
    probs = classifier.predict_proba(Xts)
    # Scoring 
    accuracy = accuracy_score(yts,predictions)
    return predictions, probs, accuracy

def validate2(classifier, Xts, yts=None, n_samples=10, params: dict = None):
    results = []
    # Check for scaling params
    if ('mean' in params.keys()) & ('std' in params.keys()):
        Xts = (Xts - params['mean'])/params['std']
    elif 'mean' in params.keys():
        Xts = Xts - params['mean']
    for i in range(Xts.shape[0]):
        samples = Xts[i]+np.random.normal(0,0.1,size=(n_samples,Xts[i].shape[0]))
        probs = classifier.predict_proba(samples)
        results.append(probs)
    return results

def random_shift(Xtr, ytr, scale=0.5, p=0.1):
    randperm = np.random.permutation(Xtr.shape[0])
    rix = []
    c1, c2, index = 0, 0, 0
    while (c1<int(Xtr.shape[0]*p*0.5)) & (c1<int(Xtr.shape[0]*p*0.5)) & (index<Xtr.shape[0]):
        if ytr[randperm[index]] == 0:
            rix.append(index)
            c1 += 1
        elif ytr[randperm[index]] == 1:
            rix.append(index)
            c2 += 1
        index += 1
    random_directions = np.random.choice([-1, 1], size=Xtr[rix].shape)
    random_sample = Xtr[rix] + scale * random_directions * np.std(Xtr, axis=0)
    Xtr, ytr = np.concatenate([Xtr, random_sample], axis=0), np.concatenate([ytr, ytr[rix]], axis=0)
    indices = np.random.permutation(Xtr.shape[0])
    return Xtr[indices], ytr[indices]

def balance_data(X, y, under=True):
    if not under:
        sampler = RandomOverSampler()
    else:
        sampler = RandomUnderSampler()
    X, y = sampler.fit_resample(X, y)
    return X, y

def fit_xgboost(param_grid, Xtr: np.array, ytr: np.array) -> XGBClassifier:
    print("Fitting XGBoost...",flush=True)

    # Initialize GridSearchCV with the classifier and parameter grid
    grid_search = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic'), param_grid=param_grid, scoring='accuracy', cv=3, verbose=3)

    # Fit the grid search to the data
    grid_search.fit(Xtr-np.mean(Xtr,axis=0), ytr)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params, flush=True)
    
    # Instantiate the classifier with the best parameters and train
    best_classifier = XGBClassifier().__class__(**best_params)
    best_classifier.fit(Xtr-np.mean(Xtr,axis=0), ytr)

    return best_classifier