"""Later review implementation; not historical employer code."""
from pathlib import Path
import argparse
import ast
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def factor_demo():
    rng=np.random.default_rng(42)
    n=80;size=rng.normal(size=n);industry=np.arange(n)%3
    X=np.column_stack([np.ones(n),size,industry==1,industry==2])
    raw=.8*size+.4*(industry==1)+rng.normal(size=n)
    residual=raw-X@np.linalg.lstsq(X,raw,rcond=None)[0]
    assert np.max(np.abs(X.T@residual))<1e-10
    return {'synthetic':True,'max_control_inner_product':float(np.max(np.abs(X.T@residual))),
            'scope':'Cross-sectional OLS neutralization illustration, NOT full Barra risk model'}

def compound_metrics(returns, periods=252):
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0 or (r <= -1).any():
        raise ValueError('No valid simple returns or return <= -100%')
    nav = np.r_[1., (1+r).cumprod().to_numpy()]
    peak = np.maximum.accumulate(nav)
    vol = r.std(ddof=1)*np.sqrt(periods)
    return {'observations':len(r), 'total_return':float(nav[-1]-1),
            'cagr':float(nav[-1]**(periods/len(r))-1),
            'max_drawdown':float((nav/peak-1).min()),
            'sharpe_rf_zero':float(r.mean()*periods/vol) if vol > 0 else None}

def warehouse_local(path):
    raw=pd.read_excel(path)
    selected=raw.iloc[:,[0,1,2]].copy()
    headers=list(selected.columns)
    # Do not silently swap the first pair: the archived code and source headers disagree.
    return {'kind':'Source schema audit; performance calculation intentionally blocked',
        'rows':len(raw),'first_three_headers':headers,
        'original_script_interpretation':['Open','Inventory','Dates'],
        'schema_match':headers==['Open','Inventory','Dates'],
        'nonpositive_first_column':int((pd.to_numeric(selected.iloc[:,0],errors='coerce')<=0).sum()),
        'decision':'Confirm instrument, price field, inventory definition and release timestamp before rerunning.'}
if __name__ == '__main__':
    assert np.isclose(compound_metrics([-.1,.1])['max_drawdown'],-.1)
    print(json.dumps(factor_demo(),indent=2))
