#!/usr/bin/env python3
"""Run any or all of the seven uploaded variants using one shared pipeline."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
import simulation_core as r

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'config.json').read_text())
SIZES = CONFIG['host_sizes']
BASE_SEED = CONFIG['base_seed']
PHASE = {'final': CONFIG['final_seed_namespace']}

def settings(variant, metric):
    return {**CONFIG['defaults'], **CONFIG['metrics'][metric],
            **CONFIG['variants'][variant], 'metric': metric}

def replacement_variation(profiles, fraction, generator):
    """Replace Binomial(richness, fraction) taxa by taxa absent before replacement.
    Draw arrivals uniformly without replacement from the same regional pool.
    Exact richness is preserved; fraction=0 preserves the original profiles.
    Each arrival receives the abundance of its paired removed taxon. Kept
    taxa retain their abundance, preserving both total mass and its variation.
    """
    if fraction == 0: return profiles.copy()
    varied=profiles.copy()
    for i,row in enumerate(profiles):
        present=np.flatnonzero(row>0); absent=np.flatnonzero(row==0)
        k=min(int(generator.binomial(len(present),fraction)),len(absent))
        removed=generator.choice(present,k,replace=False)
        incoming=generator.choice(absent,k,replace=False)
        varied[i,removed]=0.
        varied[i,incoming]=row[removed]
    return varied


def occupancy_mosaic(X,U,fraction,generator):
    """Each taxon's state independently comes from U with probability fraction.
    The endpoints are exactly X and U; no abundance threshold is introduced.
    """
    if fraction==0: return X.copy()
    if fraction==1: return U.copy()
    states=np.where(generator.random(X.shape)<fraction,U>0,X>0).astype(float)
    # Theoretical empty rows are assigned their independent donor profile.
    empty=states.sum(1)==0
    states[empty]=U[empty]>0
    return states/states.sum(1,keepdims=True)

def rng(phase,component,n,rep,*extra):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED,PHASE[phase],component,n,rep,*extra]))


def yule_tree(n,seed_rng,depth=6.):
    root=Clade(branch_length=0.,clades=[Clade(branch_length=0.),Clade(branch_length=0.)])
    active=list(root.clades)
    while len(active)<n:
        elapsed=seed_rng.exponential(1./len(active))
        for tip in active:tip.branch_length+=elapsed
        tip=active.pop(int(seed_rng.integers(len(active))))
        tip.clades=[Clade(branch_length=0.),Clade(branch_length=0.)]
        active.extend(tip.clades)
    elapsed=seed_rng.exponential(1./n)
    for tip in active:tip.branch_length+=elapsed
    tree=Tree(root=root,rooted=True)
    for i,tip in enumerate(tree.get_terminals()):tip.name=f'sp{i+1}'
    height=tree.distance(tree.root,tree.get_terminals()[0])
    for clade in tree.find_clades():clade.branch_length*=depth/height
    return tree


def parameters(setting,richness=100):
    return r.original.SimulationParams(tree_path='',n_species_pool=1000,
        root_richness=richness,max_community_size=round(1.5*richness),geom_p=.2,
        abundance_model=setting['abundance_model'],ou_alpha=setting['ou_alpha'],
        brownian_sigma=setting['sigma'],extinction_rate=.08*setting['turnover_multiplier'],
        colonization_rate=.04*setting['turnover_multiplier'],colonist_weight_multiplier=.01)


def diagnostics(tree,n,rep,phase):
    tips=tree.get_terminals();depth=np.array([tree.distance(tree.root,t) for t in tips])
    terminal=np.array([t.branch_length for t in tips]);_,D=r.original.get_patristic_distance_matrix(tree)
    return dict(phase=phase,n_hosts=n,replicate=rep+1,root_depth_mean=depth.mean(),
        root_depth_cv=depth.std()/depth.mean(),terminal_min=terminal.min(),
        terminal_mean=terminal.mean(),terminal_max=terminal.max(),
        mean_pairwise_distance=r.original.lower_triangle_values(D).mean())


def abundance_variation(profiles,sd,generator):
    """Independent host-specific abundance variation; zero taxa remain absent."""
    if sd==0:return profiles.copy()
    varied=profiles*np.exp(generator.normal(0.,sd,size=profiles.shape))
    return varied/varied.sum(axis=1,keepdims=True)


def tip_variation(profiles,setting,phase,n,rep,independent=False):
    """Apply both independently configured forms of terminal noise.

    The original noise retains seed components 17/18 for each metric; its
    newly added form uses 20/21. Neither changes tree or permutation seeds.
    Abundance noise precedes abundance-preserving taxon replacement.
    """
    offset=int(independent)
    abundance_component=(20 if setting['metric']=='BinaryDice' else 17)+offset
    replacement_component=(17 if setting['metric']=='BinaryDice' else 20)+offset
    varied=abundance_variation(profiles,setting['tip_abundance_sigma'],
                              rng(phase,abundance_component,n,rep))
    return replacement_variation(varied,setting['tip_replacement_fraction'],
                                 rng(phase,replacement_component,n,rep))


def validate_tip_noise(setting):
    sd=setting['tip_abundance_sigma']; fraction=setting['tip_replacement_fraction']
    if not np.isfinite(sd) or sd<0:
        raise ValueError('tip_abundance_sigma must be finite and nonnegative.')
    if not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('tip_replacement_fraction must be between 0 and 1.')


def task(args):
    suite,n,rep,B,setting,phase=args
    tree=yule_tree(n,rng(phase,11,n,rep),setting['tree_depth'])
    X=r.communities(tree,parameters(setting),rng(phase,13,n,rep))
    U=r.independent_profiles(tree,parameters(setting),rng(phase,14,n,rep))
    metric=setting['metric']; hc=setting['clustering']
    X=tip_variation(X,setting,phase,n,rep)
    U=tip_variation(U,setting,phase,n,rep,independent=True)
    prng=rng(phase,12,n,rep);perms=np.array([prng.permutation(n) for _ in range(B)])
    rows=[]
    if suite=='uncertainty':
        names=[t.name for t in tree.get_terminals()]
        profiles=(occupancy_mosaic(X,U,.25,rng(phase,19,n,rep,25))
                  if metric == 'BinaryDice' else .75*X+.25*U)
        conditions=[]
        for k in [0,1,3,6]:
            estimate=r.nni(tree,k,rng(phase,16,n,rep,k))
            order=r.restore_order(estimate,names)
            distance=r.original.normalized_robinson_foulds(
                r.original.get_host_splits(tree,names),r.original.get_host_splits(estimate,names))
            conditions.append((.25,profiles[order],'alternative',100,estimate,k,distance))
    elif suite=='richness':
        conditions=[(0.,tip_variation(r.communities(tree,parameters(setting,rich),rng(phase,13,n,rep)),
                    setting,phase,n,rep),
                    'alternative',rich,tree,0,0.) for rich in [10,50,100,500]]
    elif suite=='calibration':
        conditions=[(1.,U,'independent',100,tree,0,0.),
                    (0.,X[rng(phase,15,n,rep).permutation(n)],'shuffled',100,tree,0,0.)]
    else:
        levels=[0.,.25,.5,.75] # Save the full size-by-noise grid to expose condition dependence.
        conditions=[(noise,occupancy_mosaic(X,U,noise,rng(phase,19,n,rep,round(noise*100)))
                    if metric == 'BinaryDice' else (1-noise)*X+noise*U,
                    'alternative',100,tree,0,0.) for noise in levels]
    for noise,profiles,null,rich,host,k,distance in conditions:
        for row in r.evaluate(host,profiles,perms,hc=hc,metric=metric):
            row.update(suite=suite,phase=phase,n_hosts=n,replicate=rep+1,
                tree_id=f'yule_n{n}_rep{rep+1}',richness=rich,noise=noise,
                sigma=setting['sigma'],rate=setting['turnover_multiplier'],
                colonist_weight_multiplier=.01,
                tree_depth=setting['tree_depth'],
                uncertainty_kind='nni' if suite=='uncertainty' else 'none',uncertainty=k,
                realized_host_nrf=distance,hc=hc,null_model=null)
            row.update(tip_abundance_sigma=setting['tip_abundance_sigma'],
                       tip_replacement_fraction=setting['tip_replacement_fraction'])
            if setting['abundance_model'] != 'brownian':
                row.update(abundance_model=setting['abundance_model'],ou_alpha=setting['ou_alpha'])
            rows.append(row)
    return rows,diagnostics(tree,n,rep,phase)


def execute(tasks,jobs):
    rows=[];diags=[]
    with Pool(jobs) as pool:
        for i,(values,d) in enumerate(pool.imap_unordered(task,tasks),1):
            rows.extend(values);diags.append(d)
            if i%100==0 or i==len(tasks):print(f'{i}/{len(tasks)} tasks',flush=True)
    return pd.DataFrame(rows),pd.DataFrame(diags)


def checks(setting):
    validate_tip_noise(setting)
    for n in SIZES:
        t=yule_tree(n,rng('final',11,n,0),setting['tree_depth']);tips=t.get_terminals()
        assert len(tips)==n and len(set(x.name for x in tips))==n
        assert min(x.branch_length for x in tips)>0
        np.testing.assert_allclose([t.distance(t.root,x) for x in tips],setting['tree_depth'],atol=1e-12)
        names,H=r.original.get_patristic_distance_matrix(t)
        splits=r.split_matrix(r.original.get_host_splits(t,names),names)
        permutations=np.array([rng('final',90,n,i).permutation(n) for i in range(3)])
        rf,mi=r.tree_stats(splits,splits,permutations)
        rooted=r.split_matrix([frozenset(x.name for x in c.get_terminals())
            for c in t.find_clades() if c is not t.root and not c.is_terminal()],names)
        mc=r.normalized_matching_cluster(rooted,rooted,permutations)
        np.testing.assert_allclose(mc[0],0,atol=1e-12)
        assert len(rooted)==n-2 and np.all((mc>=0)&(mc<=1))
        if setting['clustering'] == 'nj':
            _,_,_,microbial=r.reconstruct_microbiota(H,names,method='nj',return_tree=True)
            mid,_,_=r.midpoint_root_for_mc(microbial)
            np.testing.assert_allclose(r.normalized_matching_cluster(
                r.rooted_clades(mid,names),rooted,permutations)[0],0,atol=1e-12)
        np.testing.assert_allclose([rf[0],mi[0]],[0,1],atol=1e-12)
    X=r.communities(t,parameters(setting),rng('final',13,n,0))
    np.testing.assert_allclose(X.sum(axis=1),1,atol=1e-12)
    return {'status':'passed','positive_edges':True,'ultrametric_depth_verified':True,
            'RF_and_information_identity_verified':True,'rooted_MC_identity_verified':True,'normalized_profiles_verified':True,
            'pilot_final_seed_namespaces_disjoint':True}


def argument_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--variants', nargs='+', choices=list(CONFIG['variants']),
                        default=list(CONFIG['variants']))
    parser.add_argument('--metrics', nargs='+', choices=list(CONFIG['metrics']),
                        default=list(CONFIG['metrics']))
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--formats', nargs='+', choices=['pdf','png','svg'],
                        default=['pdf','png','svg'])
    return parser


def save_trees(output_root, variants, count):
    # The same ensemble serves every metric and every variant with this depth.
    depths = sorted({settings(v, 'BrayCurtis')['tree_depth'] for v in variants})
    for depth in depths:
        destination = output_root/'trees'/f'depth_{depth:g}'
        destination.mkdir(parents=True, exist_ok=True)
        for n in SIZES:
            for rep in range(count):
                tree = yule_tree(n, rng('final',11,n,rep), depth)
                Phylo.write(tree,destination/f'yule_n{n}_rep{rep+1}.nwk',
                            'newick',format_branch_length='%1.15g')
    print('Shared tree ensemble saved.', flush=True)


def main():
    parser = argument_parser()
    for name, default in CONFIG['run_sizes'].items():
        parser.add_argument('--'+name.replace('_','-'), type=int, default=default)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--figures-only', action='store_true')
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--save-trees', action='store_true')
    args = parser.parse_args()
    if min(args.reps,args.null_reps,args.permutations,args.jobs) < 1:
        parser.error('Counts must be positive.')
    if args.check_only and args.figures_only:
        parser.error('Choose --check-only or --figures-only.')
    if args.no_figures and args.figures_only:
        parser.error('--no-figures conflicts with --figures-only.')
    if not args.check_only and not args.figures_only and args.null_reps <= args.reps:
        parser.error('null-reps must exceed reps for the ROC bootstrap.')
    output_root = args.output_dir.resolve()
    if args.save_trees:
        save_trees(output_root, args.variants, args.null_reps)
    for variant in args.variants:
        for metric in args.metrics:
            setting = settings(variant,metric)
            validate_tip_noise(setting)
            print(f'{variant}/{metric}', flush=True)
            if not args.figures_only:
                validation = checks(setting)
                print(validation, flush=True)
                if args.check_only:
                    continue
                destination = output_root/'results'/variant/metric
                destination.mkdir(parents=True,exist_ok=True)
                record = {'parameters':setting,'base_seed':BASE_SEED,
                          'namespace':PHASE['final'],'sizes':SIZES,
                          'reps':args.reps,'null_reps':args.null_reps,
                          'permutations':args.permutations,'jobs':args.jobs,
                          'pool':1000,'root_richness':100,'cap':150,'geom_p':.2,
                          'colonist_weight_multiplier':.01,'validation':validation}
                (destination/'run_configuration.json').write_text(json.dumps(record,indent=2)+'\n')
                start = time.time()
                for suite,count,sizes in [('main',args.reps,SIZES),
                                          ('calibration',args.null_reps,SIZES),
                                          ('richness',args.reps,[15]),
                                          ('uncertainty',args.reps,[15])]:
                    print('SUITE',suite,flush=True)
                    tasks = [(suite,n,i,args.permutations,setting,'final')
                             for n in sizes for i in range(count)]
                    frame,diag = execute(tasks,args.jobs)
                    frame = frame.sort_values(['n_hosts','replicate','noise','uncertainty','null_model','method'])
                    frame.to_csv(destination/f'{suite}_replicates.tsv',sep='\t',index=False)
                    r.summarize(frame).to_csv(destination/f'{suite}_summary.tsv',sep='\t',index=False)
                    if suite == 'calibration':
                        diag.sort_values(['n_hosts','replicate']).to_csv(
                            output_root/'results/tree_diagnostics.tsv',sep='\t',index=False)
                    print(f'elapsed {time.time()-start:.1f}s',flush=True)
            if not args.check_only and not args.no_figures:
                from make_figures import render
                render(variant,metric,output_root,args.formats)


if __name__ == '__main__':
    main()
