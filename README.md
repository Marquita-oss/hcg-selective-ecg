# HCG-SelectiveECG

Code to reproduce **Human-Certainty-Guided Selective ECG Classification: Using PTB-XL Diagnostic Likelihoods for Referral to Medical Review** (Marca, Mendoza, and Muñoz; IV CINVIE 2026).

HCG-SelectiveECG is a multitask selective-classification framework for 12-lead ECG. Instead of estimating referral uncertainty from the model's own predictive distribution (entropy, Deep Ensemble variance, MSP), it supervises an auxiliary head directly with the **diagnostic certainty recorded by the reporting cardiologist** — the continuous (0-100) likelihoods preserved in [PTB-XL](https://physionet.org/content/ptb-xl/) v1.0.3.

## What this repo contains

| Folder | Contents |
|---|---|
| `experiments/` | Six ablation/evaluation experiments (H1-H4, loss variants, sensitivity analysis) that read `master_predictions_*.csv` from a results release and produce the paper's summary tables. |
| `src/` | Multi-seed aggregation and figure-plotting utilities. |
| `posthoc/` | Post-hoc analyses (calibration by class, enrichment, quadrants) run from a released results ZIP. |
| `posthoc/real_attribution_ig/` | **Real** gradient-based attribution (Integrated Gradients, with gradient×input as a robustness check) computed on the actual trained checkpoints and raw PTB-XL signals, plus the retained-set purification analysis and the figure-generation script. This replaces an earlier, incorrect synthetic placeholder — see "A note on H4" below. |
| `paper_figures/` | Matplotlib figure generation for the manuscript. |
| `notebooks/` | Portable (no personal paths/outputs) versions of the training/analysis notebook and the figure-regeneration notebook. |
| `validate_release.py` | Checksum/manifest validation for a results release package. |

## A note on H4 (important for anyone extending this work)

An earlier version of `experiments/exp5_interpretability_attribution.py` generated its output matrix with `np.random.default_rng` and manually tuned offsets, as an illustrative placeholder — it was never connected to the trained models, and should **not** be used or re-run as-is (the file now contains an explicit warning at the top). The real computation is in `posthoc/real_attribution_ig/`: Integrated Gradients (Sundararajan et al., 2017) run independently on 5 seeds and the full test set each, with the completeness axiom checked per sample. With real attributions, the originally hypothesized QRS/precordial vs. ST-T/peripheral dissociation (H4) is **not supported** — see the paper's Results and Discussion sections for the full analysis and the honest negative result. We leave the deprecated file in place, clearly marked, for historical traceability rather than deleting it.

## Reproducing locally

The scripts need two things that are **not** included in this repository (for size and data-licensing reasons):

1. **PTB-XL raw signals** (`records100`) and the **trained checkpoints** (one `best.pt` per model per seed), referenced via the `PTBXL_PROJECT_ROOT` environment variable. PTB-XL itself is downloaded from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/); checkpoints are available on request / via the companion data release.
2. **The results release** (`03_RESULTADOS/runs/<seed>/results/tables/master_predictions_*.csv`, one set per seed), referenced via `RESULTS_ROOT`.

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

export PTBXL_PROJECT_ROOT=/path/to/ptbxl_project     # contains data/raw/ptbxl_base/records100 and runs_hcg_selective/
export RESULTS_ROOT=/path/to/results_release          # contains 03_RESULTADOS/runs/
export ATTRIBUTION_OUT_DIR=./out                       # where new CSVs/figures are written

python experiments/exp1_ablation_2x2.py
python experiments/exp2_uncertainty_baselines.py
# ... exp3-exp6

python posthoc/real_attribution_ig/compute_real_attribution.py        # gradient x input, 5 seeds
python posthoc/real_attribution_ig/compute_integrated_gradients.py    # Integrated Gradients, 5 seeds
python posthoc/real_attribution_ig/compute_retained_set_metrics.py    # retained-set purification test
python posthoc/real_attribution_ig/make_honest_figures.py             # regenerates the real attribution / retained-set / ECG-example figures
```

Five seeds (42, 13, 123, 2024, 7) are expected under `RESULTS_ROOT/03_RESULTADOS/runs/`, matching the run names used throughout the scripts (edit `SEED_RUN_MAP` in `posthoc/real_attribution_ig/compute_real_attribution.py` if your run folder names differ).

## Citation

See `CITATION.cff`. If you use this code, please cite the paper (IV CINVIE 2026 proceedings; full citation to be updated once the post-proceedings publication in Revista Ingeniare is final).

## License

MIT (see `LICENSE`) for the code in this repository. PTB-XL itself is distributed by PhysioNet under its own [Open Data Commons Attribution License](https://physionet.org/content/ptb-xl/view-license/1.0.3/); this repository does not redistribute PTB-XL data.
