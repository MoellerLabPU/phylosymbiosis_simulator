# Seven variants with both forms of terminal noise

One shared pipeline runs all seven variants with Bray–Curtis and binary
Sørensen–Dice. **Both measures now receive abundance noise and taxon replacement
noise.** Their parameters are configured independently in `config.json`:

```json
"metrics": {
  "BrayCurtis": {
    "tip_abundance_sigma": 3.1,
    "tip_replacement_fraction": 0.10
  },
  "BinaryDice": {
    "tip_abundance_sigma": 3.1,
    "tip_replacement_fraction": 0.66
  }
}
```

These settings apply to each measure across all seven variants. Set either
parameter to zero to disable that form of noise for that measure.

## Noise implementation

1. Multiply each present taxon's abundance by an independent
   `exp(Normal(0, tip_abundance_sigma))` factor, then normalize each community.
   Absent taxa remain absent.
2. Draw the number of replacements as `Binomial(present richness,
   tip_replacement_fraction)`, capped by the number of absent taxa. Choose lost
   taxa and arrivals uniformly without replacement. Arrivals are drawn from
   taxa absent before replacement, exactly as in the original Dice treatment.
3. Each arrival receives the abundance of its paired lost taxon. Kept taxa
   retain their abundance. This preserves richness, total abundance, and the
   abundance variation applied in step 1.

Both forms of noise are applied to the evolved and independent communities
in every simulation suite, including each root-richness condition. Separate
random streams keep the two terminal noise processes independent. The original
noise process for each measure retains its original seed components; the added
process uses a new component. Tree and permutation seeds are preserved.

Abundance noise changes relative abundances, but cannot change a binary Dice
distance when taxon presence is unchanged. Binary Dice still evaluates
`profiles > 0` with no abundance or detection threshold. Its original taxon-wise
occupancy mosaic is retained for the condition-noise grid. Bray–Curtis retains
its abundance admixture grid. These grid mechanisms are separate from the two
terminal noise parameters above.

## Run all seven variants and both measures

Use Python 3.11 or newer. From this folder:

```bash
python -m pip install -r requirements.txt
python run_simulations.py --jobs 4
```

Defaults are 200 alternative replicates, 1,000 null replicates, 999 permutations,
and the original seeds. The run produces results and PDF/PNG/SVG figures under
`results/<variant>/<metric>/` and `figures/<variant>/<metric>/`. Every replicate
table records **both** terminal noise parameters.

This ZIP contains the updated code and configuration. Run the command above
to generate the full results and figures for these settings. The small test
runs used to verify the implementation are documented in `verification.json`.

To select a variant and measure:

```bash
python run_simulations.py --variants brownian_0p30 --metrics BrayCurtis --jobs 4
python run_simulations.py --variants ou white_noise --metrics BinaryDice --jobs 4
python run_simulations.py --check-only
```

Variant names are `brownian_0p30`, `brownian_0p05`, `brownian_0p85`, `ou`,
`white_noise`, `neighbor_joining`, and `complete_linkage`. Metric names are
`BrayCurtis` and `BinaryDice`.

After a full run, redraw figures with `python make_figures.py`. Use
`--output-dir new_run` for a separate output folder, `--no-figures` for simulations
only, and `--formats pdf png` to select figure formats. PDFs use Helvetica;
PNG/SVG fonts depend on the installed Helvetica/Nimbus Sans fonts and fallback.
`--save-trees` optionally writes the single shared Yule-tree ensemble.

The five statistical methods, abundance evolution models, clustering methods,
and MC normalization follow the preceding consolidated implementation.
The original MIT license is included.
