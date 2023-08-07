import argparse
import sys
import pandas as pd
import os 
import argparse
import inspect
import numpy as np 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import importlib
from sklearn.calibration import CalibratedClassifierCV

def dynamic_import(module_name, function_name):
    spec = importlib.util.spec_from_file_location(module_name, f"./src/{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    return function

def import_prepare_features(modules):
    imported_functions = {}
    for module_name in modules:
        try:
            prepare_features_func = dynamic_import(module_name, "prepare_features")
            imported_functions[module_name] = prepare_features_func
        except (FileNotFoundError, AttributeError):
            print(f'Could not import prepare_features from {module_name}. Check that the file exists and contains the correct function.')
    return imported_functions
    
def get_latent(model, Xtr, ytr, Xts, params=False):
    if not params:
        model.fit(Xtr, ytr.flatten())
        latent_train = model.predict_proba(Xtr)[:,1]
        latent_test = model.predict_proba(Xts)[:,1]
    elif 'mu' in params.keys():
        model.fit(Xtr-params['mu'], ytr.flatten())
        latent_train = model.predict_proba(Xtr-params['mu'])[:,1]
        latent_test = model.predict_proba(Xts-params['mu'])[:,1]
    return latent_train, latent_test

def load_data(args):
    data = pd.read_csv(args.data).sort_values(by=['Date'],ascending=True)
    # Partition data 
    test_dates = data[(data['Date']>args.date)]['Date'].unique()
    return data, test_dates

def load_model(type: str):
    if type=='sub':
        #model = XGBClassifier(max_depth=4, n_estimators=4, objective='binary:logistic')
        base = XGBClassifier(max_depth=4, n_estimators=4, objective='binary:logistic')
        model = CalibratedClassifierCV(base, method='isotonic')
    else:
        base = XGBClassifier(max_depth=4, n_estimators=4, objective='binary:logistic')
        model = CalibratedClassifierCV(base, method='isotonic')
    return model
    
def save_scores(scores):
    print(f"Average score: {sum(scores)/len(scores):.3f}")
    if isinstance(scores, list):
        scores = np.array(scores)
    with open('scores.npy','wb') as f:
        np.save(f, scores)

def parse_models(args, root='./src'):
    if str(args.models)=='*':
        modules = sorted([str(i) for i in os.listdir(root) if '.py' in i])
    else:
        modules = sorted([str(i) for i in args.models.split(',')])
    return modules

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, help="Filename for the log", default="./log.txt")
    parser.add_argument("--data", type=str, help="Directory for the dataframe")
    parser.add_argument("--date", type=str,\
                        help="Training max date, i.e., train the model on data up to that date.")
    parser.add_argument("--models",\
                        help="List of models to use separated by a ',' (damage,weight,...)")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def calculate_aggregated_data(args, train_data, test_data, test_date):
    x_train, y_train = [], []
    x_test,  y_test  = [], []

    for ix, module in enumerate(args.modules):
        model = load_model(type='sub')
        prep_func = args.functions[module]
        if len(inspect.signature(prep_func).parameters)==3:
            (Xtr, ytr), (Xts, yts) = prep_func(train_data, test_data, test_date)
        else:
            (Xtr, ytr), (Xts, yts) = prep_func(train_data, test_data)
        if ix == 0:
            y_train, y_test = ytr, yts
        else:
            np.testing.assert_array_equal(y_train, ytr)
            np.testing.assert_array_equal(y_test, yts)
        mu = np.mean(Xtr,axis=0)
        latent_train, latent_test = get_latent(model, Xtr, ytr, Xts, params={'mu':mu})
        x_train.append(latent_train)
        x_test.append(latent_test)

    x_train, x_test = np.concatenate([i.reshape(-1,1) for i in x_train],axis=1),\
                    np.concatenate([i.reshape(-1,1) for i in x_test],axis=1) 
    return (x_train, y_train), (x_test, y_test)

def backtest(args, test_dates, data):
    scores = []
    for ix, td in enumerate(test_dates):
        train_df = data[data['Date']<td].copy()
        test_df = data[data['Date']==td].copy()
        (x_train, y_train), (x_test, y_test) = calculate_aggregated_data(args, train_df, test_df, td)
        model = load_model('f')
        model.fit(x_train, y_train.flatten())
        pred, prob = model.predict(x_test), model.predict_proba(x_test)
        ts_ix = np.argwhere(np.max(prob,axis=1)>=0.60).flatten()
        if ts_ix.shape[0]>0:
            acc = accuracy_score(pred.flatten()[ts_ix], y_test.flatten()[ts_ix])
            with open(args.log,'a') as f:
                if args.verbose == 0:
                    f.write(f"\n{td} N={ts_ix.shape[0]}/{x_test.shape[0]} Acc: {acc:.3f}")
                else:
                    f.write(f"\n{td} N={ts_ix.shape[0]}/{x_test.shape[0]} Acc: {acc:.3f}")
                    print(f"\n{td} N={ts_ix.shape[0]}/{x_test.shape[0]} Acc: {acc:.3f}",flush=True)
            scores.append(acc)
    return scores

def main():
    args = parse_args()
    args.modules = parse_models(args)
    args.functions = import_prepare_features(args.modules)
    data, test_dates = load_data(args.date)
    try:
        scores = backtest(args, test_dates, data)
        save_scores(scores)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__=="__main__":
    main()