#!/usr/bin/env python3
"""Transparent EUR/USD historical research. Does not place trades."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PIP = 0.0001
HORIZON = 12
FEATURES = ['direction','atr_pips','body_ratio','wick_up','wick_down','return_3','return_12',
            'range_position','ema_gap_atr','trend_15m','trend_1h','hour_sin','hour_cos','spread_pips']

def load(path):
    x = pd.read_csv(path)
    needed = {'timestamp','open','high','low','close'}
    if not needed.issubset(x):
        raise ValueError(f'Missing columns: {sorted(needed-set(x))}')
    x['timestamp'] = pd.to_datetime(x['timestamp'], utc=True, errors='raise')
    x = x.set_index('timestamp').sort_index()
    if x.index.has_duplicates: raise ValueError('Duplicate timestamps')
    for c in ('open','high','low','close'):
        x[c] = pd.to_numeric(x[c], errors='raise')
    if x[list(('open','high','low','close'))].isna().any().any(): raise ValueError('Missing OHLC')
    if ((x.high < x[['open','close','low']].max(axis=1)) | (x.low > x[['open','close','high']].min(axis=1))).any():
        raise ValueError('Invalid OHLC')
    # Weekend/session gaps are allowed, but any signal spanning a gap is excluded later.
    # Review gaps in the input independently; rolling indicators may bridge a weekend.
    if 'spread_pips' not in x: x['spread_pips'] = 1.0  # feature placeholder only
    x['spread_pips'] = pd.to_numeric(x['spread_pips'], errors='coerce').fillna(1.0)
    return x

def trend_completed(x, rule):
    # 5m close stamps. Resample into bars labelled by their completion time.
    y = x[['open','high','low','close']].resample(rule, label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last'})
    # A higher-timeframe bar is complete only if it contains the expected count of 5m closes.
    count = x.close.resample(rule, label='right', closed='right').count()
    expected = int(pd.Timedelta(rule) / pd.Timedelta(minutes=5))
    y.loc[count != expected, 'close'] = np.nan
    fast = y.close.ewm(span=8, adjust=False, ignore_na=True).mean()
    slow = y.close.ewm(span=21, adjust=False, ignore_na=True).mean()
    t = np.sign(fast-slow).where(y.close.notna())
    # At exactly a completed 15m/1h close, that bar is already known.
    return t.reindex(x.index, method='ffill')

def features(x):
    x = x.copy()
    prev = x.close.shift()
    tr = pd.concat([x.high-x.low, (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x['atr'] = tr.rolling(14, min_periods=14).mean()
    prior_hi = x.high.shift().rolling(12).max()
    prior_lo = x.low.shift().rolling(12).min()
    x['direction'] = np.where(x.close > prior_hi, 1, np.where(x.close < prior_lo, -1, 0))
    fast = x.close.ewm(span=8, adjust=False).mean()
    slow = x.close.ewm(span=21, adjust=False).mean()
    x['ema_gap_atr'] = (fast-slow)/x.atr
    x['trend_15m'] = trend_completed(x, '15min')
    x['trend_1h'] = trend_completed(x, '1h')
    candle_range = (x.high-x.low).replace(0,np.nan)
    x['body_ratio'] = (x.close-x.open).abs()/candle_range
    x['wick_up'] = (x.high-x[['open','close']].max(axis=1))/candle_range
    x['wick_down'] = (x[['open','close']].min(axis=1)-x.low)/candle_range
    x['return_3'] = (x.close/x.close.shift(3)-1)*10000
    x['return_12'] = (x.close/x.close.shift(12)-1)*10000
    x['range_position'] = (x.close-prior_lo)/(prior_hi-prior_lo).replace(0,np.nan)
    x['atr_pips'] = x.atr/PIP
    hour = x.index.hour+x.index.minute/60
    x['hour_sin'] = np.sin(2*np.pi*hour/24)
    x['hour_cos'] = np.cos(2*np.pi*hour/24)
    x['eligible'] = ((x.direction == x.trend_15m) & (x.direction != 0) & (x.atr>0))
    return x

def outcomes(x):
    records=[]
    gap = x.index.to_series().diff().ne(pd.Timedelta(minutes=5)).to_numpy()
    gap_count = np.cumsum(gap.astype(int))
    for i in np.flatnonzero(x.eligible.to_numpy()):
        if i+HORIZON >= len(x): continue
        if gap_count[i+HORIZON] != gap_count[i]: continue
        r = x.iloc[i]
        if not np.isfinite(r[FEATURES].to_numpy(dtype=float)).all(): continue
        direction = int(r.direction)
        entry = float(x.open.iloc[i+1]); distance = float(r.atr)
        stop = entry-direction*distance
        target = entry+direction*distance
        result=None; reason='timeout'; exit_i=i+HORIZON
        for j in range(i+1, i+HORIZON+1):
            bar=x.iloc[j]
            # If the next bar opens beyond stop, worse price applies.
            if direction*(float(bar.open)-stop) <= 0:
                result = direction*(float(bar.open)-entry)/distance
                reason='stop_gap'; exit_i=j; break
            hit_stop = bar.low <= stop if direction==1 else bar.high >= stop
            hit_target = bar.high >= target if direction==1 else bar.low <= target
            if hit_stop:
                result=-1.0; reason='stop'; exit_i=j; break
            if hit_target:
                result=1.0; reason='target'; exit_i=j; break
        if result is None:
            result=direction*(float(x.close.iloc[exit_i])-entry)/distance
        rec={'timestamp':x.index[i], 'bar_number':i, 'entry_time':x.index[i+1],
             'exit_time':x.index[exit_i], 'entry':entry, 'stop':stop, 'target':target,
             'direction':direction, 'risk_pips':distance/PIP, 'gross_R':float(result),
             'exit_reason':reason, 'target_first':int(reason=='target')}
        rec.update({f:float(r[f]) for f in FEATURES})
        records.append(rec)
    return pd.DataFrame(records)

def stats(y, p):
    if len(y)==0:return {'n':0}
    ans={'n':int(len(y)), 'target_rate':float(np.mean(y)), 'brier':float(brier_score_loss(y,p))}
    if len(np.unique(y))>1:
        ans['log_loss']=float(log_loss(y,p,labels=[0,1]))
        ans['roc_auc']=float(roc_auc_score(y,p))
    return ans

def trade_report(df, threshold, cost):
    selected=[]; last_exit=pd.Timestamp.min.tz_localize('UTC')
    for _,r in df.sort_values('timestamp').iterrows():
        if r['probability'] < threshold or r['entry_time'] <= last_exit:continue
        selected.append(r.to_dict()); last_exit=r['exit_time']
    t=pd.DataFrame(selected)
    if t.empty:return t, {'trades':0}
    t['cost_R'] = cost/t.risk_pips
    t['net_R'] = t.gross_R-t.cost_R
    t['cum_net_R'] = t.net_R.cumsum()
    gains=t.loc[t.net_R>0,'net_R'].sum()
    losses=-t.loc[t.net_R<0,'net_R'].sum()
    dd=t.cum_net_R-t.cum_net_R.cummax().clip(lower=0)
    return t, {'trades':int(len(t)), 'win_rate_net':float((t.net_R>0).mean()),
        'expectancy_net_R':float(t.net_R.mean()), 'total_net_R':float(t.net_R.sum()),
        'profit_factor':float(gains/losses) if losses>0 else None,
        'max_drawdown_R':float(-dd.min())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True)
    ap.add_argument('--out',default='reports')
    ap.add_argument('--cost-pips',type=float,default=1.0)
    ap.add_argument('--threshold',type=float,default=0.56)
    args=ap.parse_args()
    if args.cost_pips<0 or not 0<=args.threshold<=1: raise ValueError('Invalid cost or threshold')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    x=features(load(args.csv)); c=outcomes(x)
    if c.empty: raise ValueError('No candidate trades. Supply more historical data.')
    # Bar-time splits, not candidate-count splits; outcomes near boundaries excluded.
    n=len(x); a=int(n*.6); b=int(n*.8)
    train=c[(c.bar_number<a-HORIZON)]
    val=c[(c.bar_number>=a+HORIZON)&(c.bar_number<b-HORIZON)]
    test=c[(c.bar_number>=b+HORIZON)]
    if min(len(train),len(val),len(test))<20 or train.target_first.nunique()<2:
        raise ValueError(f'Insufficient independent periods/candidates: train={len(train)}, val={len(val)}, test={len(test)}. Need more data.')
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
    model.fit(train[FEATURES],train.target_first)
    val=val.copy();test=test.copy()
    val['probability']=model.predict_proba(val[FEATURES])[:,1]
    test['probability']=model.predict_proba(test[FEATURES])[:,1]
    trades, tm=trade_report(test,args.threshold,args.cost_pips)
    c['probability']=np.nan;c['split']='excluded'
    for part,name in [(train,'train'),(val,'validation'),(test,'test')]:
        c.loc[part.index,'split']=name
    c.loc[val.index,'probability']=val.probability
    c.loc[test.index,'probability']=test.probability
    c.to_csv(out/'candidates.csv',index=False)
    trades.to_csv(out/'test_trades.csv',index=False)
    metrics={'assumed_cost_pips':args.cost_pips,'fixed_probability_threshold':args.threshold,
             'total_bars':int(n),'total_candidates':int(len(c)),
             'train_candidates':int(len(train)),
             'validation':stats(val.target_first,val.probability),
             'final_test':stats(test.target_first,test.probability),
             'final_test_trades':tm,
             'warning':'OHLC and constant-cost simulation only. No proven trading edge. Never use real money based on this report.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    if len(trades):
        plt.figure(figsize=(9,4));plt.plot(range(1,len(trades)+1),trades.cum_net_R)
        plt.axhline(0,color='gray',linestyle='--');plt.xlabel('Test trades');plt.ylabel('Cumulative net R')
        plt.title('Untouched test: simulated equity');plt.tight_layout();plt.savefig(out/'equity.png');plt.close()
    bins=np.linspace(0,1,6)
    test['bin']=pd.cut(test.probability,bins,include_lowest=True)
    reliability=test.groupby('bin',observed=True).agg(pred=('probability','mean'),observed=('target_first','mean'),count=('target_first','size'))
    plt.figure(figsize=(5,5));plt.plot([0,1],[0,1],'--',color='gray')
    plt.scatter(reliability.pred,reliability.observed,s=np.maximum(20,reliability['count']*2))
    plt.xlim(0,1);plt.ylim(0,1);plt.xlabel('Predicted target-first probability');plt.ylabel('Observed frequency')
    plt.title('Untouched test calibration');plt.tight_layout();plt.savefig(out/'calibration.png');plt.close()
    print(json.dumps(metrics,indent=2));print('Saved:',out.resolve())

if __name__=='__main__':main()
