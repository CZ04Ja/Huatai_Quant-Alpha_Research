"""NEW synthetic exercise. NOT historic investment returns or employer output."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def metrics(returns, periods=250):
    r=pd.Series(returns).dropna()
    if len(r)<2 or (r<=-1).any():raise ValueError('Invalid simple-return sample')
    nav=np.r_[1.,(1+r).cumprod().values]
    return {'cagr':float(nav[-1]**(periods/len(r))-1),
            'annual_vol':float(r.std()*np.sqrt(periods)),
            'max_drawdown':float((nav/np.maximum.accumulate(nav)-1).min())}

def main():
    rng=np.random.default_rng(42)
    market=pd.Series(rng.normal(.0002,.012,1600))
    dividend=.75*market+pd.Series(rng.normal(.00005,.006,1600))
    rolling=dividend.rolling(1250).std()*np.sqrt(250)
    old=dividend.rolling(1250).std()*np.sqrt(1250)
    assert np.allclose(old.dropna()/rolling.dropna(),np.sqrt(5))
    assert np.isclose(metrics([.1,-.1])['max_drawdown'],-.1)
    assert np.isclose((1+pd.Series([.1,-.1])).prod(),.99)
    beta=dividend.cov(market)/market.var()
    result={'label':'SYNTHETIC LEARNING ONLY; NOT HISTORICAL PERFORMANCE',
            'seed':42,'simulated_dividend':metrics(dividend),
            'simulated_market':metrics(market),'beta':float(beta),
            'old_vol_overstatement_factor':float(np.sqrt(5)),
            'tests':'compound return, initial NAV drawdown, annualization PASS'}
    (Path(__file__).parent/'dividend_synthetic_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
