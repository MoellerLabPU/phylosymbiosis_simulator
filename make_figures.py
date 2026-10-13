#!/usr/bin/env python3
"""Make Figure 1 and Figures S1–S13 from the saved simulations."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader
import plot_utils as f

from run_simulations import HERE, CONFIG, settings, argument_parser

def render(variant, metric, output_root=HERE, formats=('pdf','png','svg')):
    figure_name = CONFIG['figure_names'][variant][metric]
    R = Path(output_root)/'results'/variant/metric
    OUT = Path(output_root)/'figures'/variant/metric
    OUT.mkdir(parents=True,exist_ok=True)
    f.RESULTS = R
    f.LABELS['cophenetic'] = ('NJ-distance Mantel' if settings(variant,metric)['clustering'] == 'nj'
                              else 'Cophenetic Mantel')
    METHODS, COLORS, LABELS = f.METHODS, f.COLORS, f.LABELS
    def load(suite, kind='summary'):
        return pd.read_csv(R/f'{suite}_{kind}.tsv',sep='\t')
    data = load('main'); reps = load('main', 'replicates')
    cal = load('calibration'); rich = load('richness'); unc = load('uncertainty')
    f.auc_table(reps)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 8))
    fig.subplots_adjust(left=.073, right=.975, top=.935, bottom=.145, wspace=.38, hspace=.52)

    ax = axes[0, 0]
    f.lines(ax, rich, 'richness'); f.panel(ax, 'A', 'Community-richness sensitivity')
    ax.set_xscale('log'); ax.set_xticks([10, 50, 100, 500], [10, 50, 100, 500])
    ax.set_xlabel('Root microbiota richness'); ax.set_ylabel('Detection rate')

    ax = axes[0, 1]
    for null, style in [('independent', '-'), ('shuffled', '--')]:
        f.lines(ax, cal[cal.null_model == null], 'n_hosts', ci=False, style=style, label=False)
    ax.axhline(.05, color='#535963', ls=':', lw=1)
    ax.set_ylim(0, max(.15, float(cal.proportion.max()) * 1.4))
    ax.set_yticks([0, .05, .10, .15]); ax.set_xticks([8, 15, 30, 50])
    f.panel(ax, 'B', 'False positives on Yule trees')
    ax.set_xlabel('Number of host tips'); ax.set_ylabel('False-positive rate')

    ax = axes[0, 2]
    subset = pd.concat([data[data.n_hosts == 15],
                        cal[(cal.n_hosts == 15) & (cal.null_model == 'independent')]])
    subset.to_csv(R / f'{figure_name}C_plot_data.tsv', sep='\t', index=False)
    f.lines(ax, subset, 'noise'); f.panel(ax, 'C', 'Detection rate under increasing admixture')
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xlabel('Probability of independent taxon state' if metric == 'BinaryDice'
                                                        else 'Independent-community fraction')
    ax.set_ylabel('Detection rate')

    ax = axes[1, 0]
    subset = data[data.noise == 0]
    subset.to_csv(R / f'{figure_name}D_plot_data.tsv', sep='\t', index=False)
    f.lines(ax, subset, 'n_hosts'); f.panel(ax, 'D', 'Host-tip number: no admixture')
    ax.set_xticks([8, 15, 30, 50]); ax.set_xlabel('Number of host tips'); ax.set_ylabel('Detection rate')
    a = rich[rich.richness == 100].set_index('method')
    d = subset[subset.n_hosts == 15].set_index('method')
    np.testing.assert_allclose(a.loc[METHODS].proportion, d.loc[METHODS].proportion, atol=0)

    ax = axes[1, 1]; cal_reps = load('calibration', 'replicates')
    for method in METHODS:
        null = cal_reps[(cal_reps.n_hosts == 15) & (cal_reps.null_model == 'independent')
                        & (cal_reps.method == method)].p.to_numpy()
        alternative = reps[(reps.n_hosts == 15) & (reps.noise == .5)
                           & (reps.method == method)].p.to_numpy()
        fp, tp = f.roc(null, alternative)
        ax.plot(fp, tp, color=COLORS[method], linestyle=f.STYLES[method],
                marker=f.MARKERS[method], ms=3, markevery=max(1,len(fp)//10))
    ax.plot([0, 1], [0, 1], color='#8D9299', ls=':', lw=.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    f.panel(ax, 'E', 'ROC at 50% admixture')
    ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')

    ax = axes[1, 2]
    f.lines(ax, unc, 'uncertainty'); f.panel(ax, 'F', 'Host-topology perturbation')
    ax.set_xticks([0, 1, 3, 6]); ax.set_xlabel('Attempted NNI moves'); ax.set_ylabel('Detection rate')
    f.legend(fig, .035)
    for extension in formats:
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=300)
        destination = OUT / f'{figure_name}.{extension}'
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(destination)
    plt.close(fig)
    if 'pdf' in formats: assert len(PdfReader(OUT/f'{figure_name}.pdf').pages) == 1
    print(f'{variant}/{metric}: {figure_name} saved; ROC and panel anchors verified.')


def main():
    args = argument_parser(__doc__).parse_args()
    for variant in args.variants:
        for metric in args.metrics:
            render(variant,metric,args.output_dir.resolve(),args.formats)


if __name__ == '__main__':
    main()

