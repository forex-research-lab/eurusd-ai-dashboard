"""Streamlit UI for the EUR/USD historical research prototype. No execution or live feed."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from research import load, features, outcomes, FEATURES, HORIZON, stats, trade_report

st.set_page_config(page_title="EUR/USD Research Lab", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {padding-top:1.1rem; padding-left:0.8rem; padding-right:0.8rem; max-width:1100px;}
[data-testid="stMetric"] {background:rgba(125,125,125,.08);border-radius:12px;padding:10px;}
@media (max-width: 640px) {
  .block-container {padding-top:.5rem; padding-left:.55rem; padding-right:.55rem;}
  h1 {font-size:1.5rem!important;line-height:1.25!important;}
  h2 {font-size:1.15rem!important;}
  [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  [data-testid="column"] {min-width:45%!important;flex:1 1 45%!important;}
  [data-testid="stMetricValue"] {font-size:1.35rem!important;}
  [data-testid="stFileUploader"] {border-radius:12px;}
}
</style>""", unsafe_allow_html=True)
st.title("EUR/USD · AI Research")
st.caption("Historical analysis only · 5-minute candles · demo research · no broker connection or order execution")
st.warning("Experimental model, not a validated trading strategy. Historical or synthetic results do not predict profits.")

with st.sidebar:
    st.header("Research settings")
    source=st.radio("Data source",["Upload EUR/USD CSV","Synthetic demonstration"],index=0)
    cost=st.number_input("Assumed all-in round-trip cost (pips)",0.0,20.0,1.0,0.1)
    threshold=st.slider("Illustrative probability filter",0.30,0.90,0.56,0.01)
    st.caption("Changing the threshold after viewing test outcomes makes those outcomes exploratory, not an untouched test.")
    chart_bars=st.slider("Candles on chart",60,500,120,20)
    st.divider()
    st.markdown("**Required CSV:** `timestamp,open,high,low,close`")
    st.caption("5-minute bars, timestamps at bar close, UTC preferred. Optional `spread_pips`.")
    st.caption("Costs are a fixed simulation assumption; uploaded spread is a model feature, not automatically an execution-cost series.")

uploaded=None
if source=="Upload EUR/USD CSV":
    uploaded=st.file_uploader("Upload historical EUR/USD 5-minute OHLC CSV",type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to run the model. Alternatively choose Synthetic demonstration to explore the interface.")
        st.stop()
    file_bytes=uploaded.getvalue()
else:
    demo=Path(__file__).parent/"data"/"synthetic_TEST_ONLY.csv"
    if not demo.exists():
        st.error("Bundled synthetic demonstration data not found.")
        st.stop()
    file_bytes=demo.read_bytes()
    st.error("SYNTHETIC DATA — interface demonstration only. These prices are fabricated and results have no market significance.")

@st.cache_data(show_spinner=False)
def compute(raw_bytes):
    with tempfile.NamedTemporaryFile(suffix=".csv",delete=False) as f:
        f.write(raw_bytes)
        temp_path=f.name
    try:
        bars=load(temp_path)
        x=features(bars)
        candidates=outcomes(x)
        if candidates.empty:
            raise ValueError("No valid breakout candidates. Try a longer, complete 5-minute dataset.")
        n=len(x); a=int(n*.6); b=int(n*.8)
        train=candidates[candidates.bar_number<a-HORIZON].copy()
        val=candidates[(candidates.bar_number>=a+HORIZON)&(candidates.bar_number<b-HORIZON)].copy()
        test=candidates[candidates.bar_number>=b+HORIZON].copy()
        if min(len(train),len(val),len(test))<20 or train.target_first.nunique()<2:
            raise ValueError(f"Not enough independent candidates: train={len(train)}, validation={len(val)}, test={len(test)}. Supply more data.")
        model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                            LogisticRegression(max_iter=1000,random_state=42))
        model.fit(train[FEATURES],train.target_first)
        val["probability"]=model.predict_proba(val[FEATURES])[:,1]
        test["probability"]=model.predict_proba(test[FEATURES])[:,1]
        all_candidates=candidates.copy()
        all_candidates["probability"]=np.nan
        all_candidates["split"]="excluded"
        for part,name in [(train,"train"),(val,"validation"),(test,"test")]:
            all_candidates.loc[part.index,"split"]=name
        all_candidates.loc[val.index,"probability"]=val.probability
        all_candidates.loc[test.index,"probability"]=test.probability
        return x,all_candidates,val,test
    finally:
        Path(temp_path).unlink(missing_ok=True)

try:
    with st.spinner("Building features, fitting the model and evaluating unseen history..."):
        bars,candidates,val,test=compute(file_bytes)
        trades,trade_stats=trade_report(test,threshold,cost)
except Exception as exc:
    st.error(f"Unable to analyse this file: {exc}")
    st.stop()

tab1,tab2,tab3,tab4=st.tabs(["Overview & chart","Probability validation","Backtest & costs","Data & downloads"])
with tab1:
    a,b,c,d=st.columns(4)
    a.metric("5-minute bars",f"{len(bars):,}")
    b.metric("Candidate setups",f"{len(candidates):,}")
    c.metric("Final-test candidates",f"{len(test):,}")
    d.metric("Selected test trades",f"{len(trades):,}")
    st.subheader("Historical EUR/USD candles")
    view=bars.tail(chart_bars)
    fig=go.Figure(go.Candlestick(x=view.index,open=view.open,high=view.high,low=view.low,close=view.close,
                                 name="EUR/USD"))
    visible=candidates[(candidates.timestamp>=view.index.min())&(candidates.timestamp<=view.index.max())]
    if len(visible):
        fig.add_trace(go.Scatter(x=visible.timestamp,y=visible.entry,mode="markers",
                                 marker=dict(symbol=["triangle-up" if v==1 else "triangle-down" for v in visible.direction],
                                             size=10,color=["#16a34a" if v==1 else "#dc2626" for v in visible.direction]),
                                 name="Candidate entry (next open)"))
    fig.update_layout(height=370,xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=25,b=5),
                      xaxis_title="UTC",yaxis_title="EUR/USD")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Markers show historical candidate entries, not trade recommendations. The probability model is evaluated on later data only.")
    st.subheader("Recent out-of-sample candidates")
    cols=["timestamp","direction","probability","target_first","exit_reason","risk_pips"]
    st.dataframe(candidates[candidates.split.isin(["validation","test"])][cols].tail(15).sort_values("timestamp",ascending=False),
                 use_container_width=True,hide_index=True)

with tab2:
    st.subheader("Probability quality on chronological test period")
    pstats=stats(test.target_first,test.probability)
    a,b,c=st.columns(3)
    a.metric("Brier score",f"{pstats['brier']:.3f}")
    b.metric("Observed target-first rate",f"{pstats['target_rate']:.1%}")
    c.metric("ROC AUC",f"{pstats.get('roc_auc',float('nan')):.3f}")
    edges=np.linspace(0,1,6)
    cal=test.assign(bucket=pd.cut(test.probability,edges,include_lowest=True))
    cal=cal.groupby("bucket",observed=True).agg(predicted=("probability","mean"),
        observed=("target_first","mean"),n=("target_first","size")).reset_index()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=cal.predicted,y=cal.observed,mode="lines+markers",
                             name="Model",text=cal.n,hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>Count %{text}"))
    fig.update_layout(height=340,xaxis_title="Average predicted probability",yaxis_title="Observed target-first frequency",
                      xaxis_range=[0,1],yaxis_range=[0,1])
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Target-first probability does not equal probability of positive net P&L: timeout outcomes and costs matter.")
    st.dataframe(cal,use_container_width=True,hide_index=True)

with tab3:
    st.subheader("Simulated final-test results")
    a,b,c,d=st.columns(4)
    a.metric("Trades",trade_stats.get("trades",0))
    b.metric("Net expectancy",f"{trade_stats.get('expectancy_net_R',0):.3f} R/trade" if len(trades) else "N/A")
    c.metric("Total net result",f"{trade_stats.get('total_net_R',0):.2f} R" if len(trades) else "N/A")
    d.metric("Max drawdown",f"{trade_stats.get('max_drawdown_R',0):.2f} R" if len(trades) else "N/A")
    if len(trades):
        fig=go.Figure(go.Scatter(x=trades.exit_time,y=trades.cum_net_R,mode="lines",name="Net cumulative R"))
        fig.add_hline(y=0,line_dash="dash")
        fig.update_layout(height=330,xaxis_title="Exit date (UTC)",yaxis_title="Cumulative net R")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(trades[["timestamp","direction","probability","entry","stop","target","exit_reason",
                             "gross_R","cost_R","net_R"]].tail(30).sort_values("timestamp",ascending=False),
                     use_container_width=True,hide_index=True)
    else:
        st.info("No test trades passed this probability filter.")
    st.warning("Changing cost or threshold here is exploratory sensitivity analysis. Do not call the displayed result an untouched evaluation after selecting settings based on its performance.")

with tab4:
    st.subheader("Data quality and reproducibility")
    intervals=bars.index.to_series().diff().dropna()
    gaps=int((intervals!=pd.Timedelta(minutes=5)).sum())
    st.write(f"**Date range (UTC):** {bars.index.min()} to {bars.index.max()}")
    st.write(f"**Non-5-minute intervals:** {gaps} (weekend/session closures may explain some gaps)")
    st.write(f"**Split:** 60% training / 20% validation / 20% final test by chronological bar position; 12-bar embargo.")
    st.write("**Prediction:** target hit first within 12 future five-minute bars. Stop-first and timeouts count as not-target.")
    st.write("**Execution approximation:** next-bar open; 1 ATR stop and target; stop priority when both levels touched within a candle.")
    metrics={
        "research_only":True,"synthetic_data":source=="Synthetic demonstration",
        "date_start":str(bars.index.min()),"date_end":str(bars.index.max()),
        "bars":len(bars),"candidates":len(candidates),"validation":stats(val.target_first,val.probability),
        "test":stats(test.target_first,test.probability),"backtest":trade_stats,
        "round_trip_cost_pips_assumed":cost,"probability_filter_exploratory":threshold,
        "caution":"No demonstrated trading edge. Results are sensitive to execution assumptions and selection."
    }
    st.download_button("Download metrics JSON",json.dumps(metrics,indent=2),"metrics.json","application/json")
    st.download_button("Download all candidates CSV",candidates.to_csv(index=False),"candidates.csv","text/csv")
    st.download_button("Download selected test trades CSV",trades.to_csv(index=False),"test_trades.csv","text/csv")
    st.caption("This dashboard runs locally in your browser using Streamlit. No orders are submitted.")
