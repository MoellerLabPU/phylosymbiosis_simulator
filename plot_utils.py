#!/usr/bin/env python3
"""Plotting and ROC helpers for Figure 1."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
# Register the Helvetica-compatible fallback when available on Linux.
for font in Path('/usr/share/fonts/opentype/urw-base35').glob('NimbusSans-*.otf'):
    font_manager.fontManager.addfont(str(font))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
# Five colors from the Okabe-Ito palette, reinforced by shape and line pattern.
COLORS={'raw':'#0072B2','cophenetic':'#D55E00','rf':'#009E73','mci':'#CC79A7','mc':'#E69F00'}
MARKERS={'raw':'o','cophenetic':'s','rf':'^','mci':'D','mc':'v'}
STYLES={'raw':'-','cophenetic':'--','rf':'-.','mci':':','mc':(0,(5,1,1,1))}
LABELS={'raw':'Raw Mantel','cophenetic':'Cophenetic Mantel','rf':'Normalized RF','mci':'Clustering information','mc':'Normalized MC'}
METHODS=list(COLORS)
plt.rcParams.update({'font.family':['Helvetica', 'Nimbus Sans', 'sans-serif'], 'pdf.use14corefonts':True,'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.linewidth':.75,'xtick.major.width':.7,'ytick.major.width':.7,'lines.linewidth':1.6})

def lines(ax,frame,x,ci=True,style=None,label=True):
    for m in METHODS:
        d=frame[frame.method==m].sort_values(x)
        if d.empty:continue
        ax.plot(d[x],d.proportion,color=COLORS[m],marker=MARKERS[m],ms=3.8,
                linestyle=STYLES[m] if style is None else style,label=LABELS[m] if label else None)
        if ci:ax.fill_between(d[x],d.ci_low,d.ci_high,color=COLORS[m],alpha=.1,lw=0)
    ax.set_ylim(0,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis='y',color='#E5E7EA',lw=.55,zorder=0)

def panel(ax,letter,title):
    ax.text(-.17,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=13,va='bottom')

def legend(fig,y=.025):
    handles=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],linestyle=STYLES[m],
                    ms=4,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=5,frameon=False,columnspacing=1.8,handlelength=2)

def roc(null,alt):
    # Lower permutation p means stronger evidence. Include each tied score
    # group together; straight segments between vertices give half credit to
    # ties, matching auc_pair. steps-post incorrectly gives no credit to ties.
    null=np.asarray(null);alt=np.asarray(alt)
    thresholds=np.unique(np.r_[null,alt])
    fpr=np.r_[0.,[(null<=t).mean() for t in thresholds]]
    tpr=np.r_[0.,[(alt<=t).mean() for t in thresholds]]
    np.testing.assert_allclose(np.trapezoid(tpr,fpr),auc_pair(null,alt),atol=1e-12)
    return fpr,tpr

def auc_pair(null,alt):
    ordered=np.sort(null)
    lower=np.searchsorted(ordered,alt,side='left')
    upper=np.searchsorted(ordered,alt,side='right')
    return float(np.mean((len(null)-upper+.5*(upper-lower))/len(null)))


def auc_table(df):
    calibration=pd.read_csv(RESULTS/'calibration_replicates.tsv',sep='\t')
    rows=[];curves=[]
    for m in METHODS:
        null=calibration[(calibration.n_hosts==15)&(calibration.null_model=='independent')
                         &(calibration.method==m)].sort_values('replicate').p.to_numpy()
        alt=df[(df.n_hosts==15)&(df.noise==.5)&(df.null_model=='alternative')
               &(df.method==m)].sort_values('replicate').p.to_numpy()
        value=auc_pair(null,alt);rng=np.random.default_rng(991+15+50+METHODS.index(m))
        bs=[]
        for _ in range(500):
            sampled=rng.integers(len(alt),size=len(alt))
            extra=rng.choice(null[len(alt):],len(null)-len(alt),replace=True)
            bs.append(auc_pair(np.r_[null[sampled],extra],alt[sampled]))
        lo,hi=np.quantile(bs,[.025,.975])
        rows.append(dict(n_hosts=15,noise=.5,null_model='independent',method=m,
            auc=value,ci_low=lo,ci_high=hi,n_null=len(null),n_alternative=len(alt),bootstrap_replicates=500))
        fp,tp=roc(null,alt)
        for a,b in zip(fp,tp):curves.append(dict(method=m,false_positive_rate=a,true_positive_rate=b))
    pd.DataFrame(rows).to_csv(RESULTS/'roc_auc.tsv',sep='\t',index=False)
    pd.DataFrame(curves).to_csv(RESULTS/'roc_curves.tsv',sep='\t',index=False)
    return pd.DataFrame(rows)
